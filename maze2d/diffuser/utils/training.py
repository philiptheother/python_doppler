import os
import sys
import copy
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import einops
import pdb

from diffuser.datasets import load_environment, SequenceDataset, LocationDataset

from .arrays import batch_to_device, to_np, to_torch, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from .serialization import load_diffusion

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
        output=print,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.output = output
        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, n_valid_steps=None):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                self.output(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            self.step += 1

    def test(self, epoch, bs=1, n_avg=5, vis_freq=10):
        self.output("[ utils/training ] Testing...")
        env = load_environment(self.dataset.env.name)

        if isinstance(self.dataset, LocationDataset):
            from diffuser.guides import CondPolicy
            cond_dim = 2
            policy = CondPolicy(cond_dim, diffusion_model=self.ema_model, normalizer=self.dataset.normalizer)
        elif isinstance(self.dataset, SequenceDataset):
            from diffuser.guides import Policy
            cond_dim = self.dataset.observation_dim
            policy = Policy(self.ema_model, self.dataset.normalizer)
        else:
            raise NotImplementedError

        #---------------------------------- main loop ----------------------------------#
        all_score, all_t, all_R, all_d = np.zeros(n_avg), np.zeros(n_avg), np.zeros(n_avg), np.zeros(n_avg)
        for idx in range(n_avg):

            observation = env.reset()
            ## observations for rendering
            plan = []
            rollout = [observation.copy()]

            total_reward = 0
            t = 0
            while t < env.max_episode_steps:
                for h in range(self.dataset.horizon-1):
                    state = env.state_vector().copy()
                    if h == 0:
                        cond = {0: observation,}
                        action, samples, _, _ = policy(cond, batch_size=bs)
                        actions = samples.actions[0]
                        sequence = samples.observations[0]
                        plan.append(sequence)

                    if h < len(sequence) - 1:
                        next_waypoint = sequence[h+1]
                    else:
                        next_waypoint = sequence[-1].copy()
                        next_waypoint[2:] = 0

                    ## can use actions or define a simple controller based on state predictions
                    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

                    next_observation, reward, terminal, _ = env.step(action)
                    total_reward += reward
                    score = env.get_normalized_score(total_reward)

                    ## update rollout observations
                    rollout.append(next_observation.copy())

                    t += 1
                    if terminal or t >= env.max_episode_steps:
                        break

                    observation = next_observation

            # save plan and rollout
            savepath = os.path.join(self.logdir, 'test', f'{epoch}_{idx}_plan_rollout.png')
            self.renderer.composite(savepath, (np.concatenate(plan, axis=0), np.array(rollout)), ncol=2)

            # logger.finish(t, env.max_episode_steps, score=score, value=0)
            all_score[idx] = score
            all_t[idx] = t
            all_R[idx] = total_reward
            all_d[idx] = terminal

        output_str = f'[ utils/training ] Test R avg: {all_R.mean():.2f} | score avg: {all_score.mean():.4f} | {action} | '
        self.output(output_str)

        ## save result as a json file
        json_path = os.path.join(self.logdir, 'rollout.json')
        json_data = {'epoch': epoch, 'score': all_score.tolist(), 'step': all_t.tolist(), 'return': all_R.tolist(), 'term': all_d.tolist()}
        json.dump(json_data, open(json_path, 'a'), indent=2, sort_keys=True)

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        self.output(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''
        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, 'sample', f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):
            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## get trajectories and condition at t=0 from batch
            trajectories = to_np(batch.trajectories)

            ## [ batch_size x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]
            refs = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, 'sample', f'{self.step}-{i}-ref-sample.png')
            self.renderer.composite(savepath, np.concatenate([refs, observations]), ncol=5)


class RLTrainer(object):
    def __init__(
        self,
        dataset,
        renderer,
        option_horizon,  # number of transition timesteps inside the option
        discount,
        model_config,
        diffusion_loadpath=None,
        train_batch_size=32,
        train_lr=0.001,
        a_n_n_target=16,
        a_n_noise=True,
        policy_noise=0.02,
        noise_clip=0.05,
        update_mode="soft",
        policy_freq=2,
        tau=0.005,
        ddqn=True,

        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        results_folder='./results',
        n_reference=10,
        n_episodes=20,
        valid_freq=None,
        output=print,
    ):
        super().__init__()
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))

        self.renderer = renderer
        
        self.option_horizon = option_horizon
        self.max_option = 1.0  # for normalized trajectories
        self.discount = discount

        assert diffusion_loadpath is not None, "Diffusion loadpath must be provided"
        diffusion_experiment = load_diffusion('logs', dataset.env.name, diffusion_loadpath)
        self.diffusion = diffusion_experiment.ema
        if isinstance(diffusion_experiment.dataset, LocationDataset):
            self.cond_dim = diffusion_experiment.dataset.observation_dim - 2
        elif isinstance(diffusion_experiment.dataset, SequenceDataset):
            self.cond_dim = diffusion_experiment.dataset.observation_dim
        else:
            raise NotImplementedError

        self.network = model_config(output=output)
        self.network2 = model_config(output=output)
        self.target_network = model_config(output=output)
        self.target_network2 = model_config(output=output)
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.weight_transfer(from_model=self.network2, to_model=self.target_network2)      

        self.train_lr = train_lr
        self.start_train_lr = train_lr
        self.optimizer = torch.optim.Adam([{'params': self.network.parameters()}, {'params': self.network2.parameters()}], lr=self.train_lr)

        self.a_n_n_target = a_n_n_target
        self.a_n_noise = a_n_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_counter = 0
        self.update_mode = update_mode
        self.policy_freq = policy_freq
        self.tau = tau
        self.ddqn = ddqn

        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.output = output
        self.logdir = results_folder
        self.n_reference = n_reference
        self.n_episodes = n_episodes

        self.step = 0

        self.evaluator = self.get_evaluator()

    def get_c_max_o(self, q, s, ltl, o, obs, batch_size, n_o, **sample_kwargs):
        o_input = torch.flatten(o, start_dim=2)
        o_input = torch.reshape(o_input, (batch_size*n_o,-1))

        s_clones = s.unsqueeze(1).expand(-1,n_o,-1)
        s_clones = torch.reshape(s_clones, (batch_size*n_o,-1))

        ltl_clones = np.repeat(ltl[:,None,:],n_o,axis=1)
        ltl_clones = ltl_clones.reshape((batch_size*n_o,-1))

        q_values = q(torch.cat((s_clones, o_input), dim=1), ltl_clones)
        q_values = torch.reshape(q_values, (batch_size, n_o,-1))
        q_values = q_values.squeeze(2)
        
        elite_ind = q_values.argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=o.device)
        elite_o = o[batch_indices, elite_ind]
        if 'return_obs' in sample_kwargs and sample_kwargs['return_obs']:
            assert obs is not None, "obs must be provided to return elite observations"
            elite_obs = obs[batch_indices, elite_ind]
            return elite_o, elite_obs
        else:
            return elite_o
        
    def get_c_max_o_diffusion(self, q, s, ltl, diffusion, cond_dim, n_o=16, guide=None, **sample_kwargs):
        batch_size = s.shape[0]
        assert cond_dim <= s.shape[1], "provided state dim must be greater than or equal to the diffusion condition dim"
        assert q.state_dim <= s.shape[1], "provided state dim must be greater than or equal to the q network state dim"
        
        conditions = {0: s[:,:cond_dim].repeat(n_o,1)}
        if guide is not None:
            sample = diffusion(conditions, guide=guide, verbose=False, **sample_kwargs)
            sample = sample.trajectories
        else:
            sample = diffusion(conditions, verbose=False)

        o_b_c = sample[:, 1:, self.dataset.action_dim:self.dataset.action_dim+q.state_dim]
        o = [o_b_c[range(i,len(o_b_c),batch_size)] for i in range(batch_size)]
        o = torch.stack(o, dim=0)
        if 'return_obs' in sample_kwargs and sample_kwargs['return_obs']:
            obs_b_c = sample[:, 1:, self.dataset.action_dim:]
            obs = [obs_b_c[range(i,len(obs_b_c),batch_size)] for i in range(batch_size)]
            obs = torch.stack(obs, dim=0)
        else:
            obs = None
        return self.get_c_max_o(q, s[:,:q.state_dim], ltl, o, obs, batch_size, n_o, **sample_kwargs)

    def batch_rl(self, trj, s, o, r, s_n, d, ids_trj, ids_ltl):
        ids_trj = ids_trj.cpu().numpy().astype(int)
        ids_ltl = ids_ltl.cpu().numpy().astype(int)
        ltls_goal_graph = self.dataset.graphs[ids_ltl]
        idx_ltl_next = [self.dataset.idx_ltl_next[idx_trj, idx_ltl] for idx_trj, idx_ltl in zip(ids_trj,ids_ltl)]
        ltls_next_graph = self.dataset.graphs[np.array(idx_ltl_next, dtype=int)]

        state_dim = self.network.state_dim
        q1 = self.network(torch.cat((s[:,:state_dim], torch.flatten(o,start_dim=1)), dim=1), ltls_goal_graph)
        q2 = self.network2(torch.cat((s[:,:state_dim], torch.flatten(o,start_dim=1)), dim=1), ltls_goal_graph)

        def get_target_constrained():
            with torch.no_grad():
                o_n = self.get_c_max_o_diffusion(q = self.target_network,
                                                 s = s_n,
                                                 ltl = ltls_next_graph,
                                                 diffusion = self.diffusion,
                                                 cond_dim = self.cond_dim,
                                                 n_o = self.a_n_n_target)

                if self.a_n_noise:
                    noise = (torch.randn_like(o_n) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    o_n = (o_n + noise).clamp(-self.max_option, self.max_option)
                input = torch.cat((s_n[:,:state_dim], torch.flatten(o_n,start_dim=1)), dim=1)
                if self.ddqn:
                    q1_n = self.target_network(input, ltls_next_graph).detach()
                    q2_n = self.target_network2(input, ltls_next_graph).detach()
                    q_n = torch.min(q1_n, q2_n)
                else:
                    q_n = self.target_network(input, ltls_next_graph).detach()
            return (1.0-d) * q_n

        bellman_target = r + self.discount**self.option_horizon * get_target_constrained()
        if self.ddqn:
            loss = F.mse_loss(q1.squeeze(), bellman_target.squeeze()) + F.mse_loss(q2.squeeze(), bellman_target.squeeze())
        else:
            loss = F.mse_loss(q1.squeeze(), bellman_target.squeeze())
        return loss

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            batch = next(self.dataloader)
            batch = batch_to_device(batch)

            loss = self.batch_rl(*batch)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # updating target network
            if self.update_counter == self.policy_freq-1:
                self.update_counter = 0
                if "hard" == self.update_mode:
                    self.weight_transfer(from_model=self.network, to_model=self.target_network)
                    if self.ddqn:
                        self.weight_transfer(from_model=self.network2, to_model=self.target_network2)
                elif "soft" == self.update_mode:
                    self.soft_update(source=self.network, target=self.target_network)
                    if self.ddqn:
                        self.soft_update(source=self.network2, target=self.target_network2)
                else:
                    raise NotImplementedError
            else:
                self.update_counter += 1

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                self.output(f'{self.step}: loss {loss:8.4f} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(self.n_episodes)

            self.step += 1

    def soft_update(self, source, target):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param source: Source network (PyTorch)
        :param target: Target network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau*param.data + (1.0-self.tau)*target_param.data)

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def save(self, epoch):
        '''
            saves model and optimizer to disk;
        '''
        data = {
            'step': self.step,
            'q1': self.network.state_dict(),
            'q2': self.network2.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        self.output(f'[ utils/training ] Saved model to {savepath}')

    def load(self, epoch):
        '''
            loads model and optimizer from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.network.load_state_dict(data['q1'])
        self.network2.load_state_dict(data['q2'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.target_network = copy.deepcopy(self.network)
        self.target_network2 = copy.deepcopy(self.network2)

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''
        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, 'sample', f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, n_episodes=20):
        idx = random.randint(0, len(self.dataset.indices)-1)
        path_ind, start, end = self.dataset.indices[idx]
        states = self.dataset.fields.normed_observations[path_ind, start]
        states = to_torch(states, device='cuda:0').unsqueeze(0)
        observations_all = []

        ltl_goal_tup = random.choice(self.dataset.ltls_goal_tup)
        idx_ltl_goal_tup = self.dataset.ltls_goal_tup.index(ltl_goal_tup)
        ltl_goal_str = self.dataset.ltls_goal_str[idx_ltl_goal_tup]

        ltl_tup = ltl_goal_tup
        success = False
        for i in range(n_episodes):
            idx_ltl_all_tup = self.dataset.ltls_all_tup.index(ltl_tup)
            ltl_goal_graph = self.dataset.graphs[idx_ltl_all_tup]
            ltls_goal_graph = ltl_goal_graph[None,:]

            option = self.get_c_max_o_diffusion(q = self.network,
                                                s = states,
                                                ltl = ltls_goal_graph,
                                                diffusion = self.diffusion,
                                                cond_dim = self.cond_dim,
                                                n_o=self.a_n_n_target)

            trj, observations, states = self.option2trj(option)
            observations_all.append(observations)
            assignments = self.evaluator.get_assignments(trj)
            assignments = self.dataset.boundary2prop(assignments[0].cpu().numpy())
            for a in assignments:
                ltl_tup = self.dataset.progression(ltl_tup, a)
                # Computing the LTL reward and done signal
                ltl_done = False
                if ltl_tup == 'True':
                    ltl_done = True
                    success = True
                elif ltl_tup == 'False':
                    ltl_done = True

                if ltl_done:
                    break
            if ltl_done:
                break

        savepath = os.path.join(self.logdir, 'sample', f'{self.step}-{success}-{ltl_goal_str}.png')
        self.renderer.composite(savepath, np.concatenate(observations_all), ncol=1)
        return np.concatenate(observations_all)

    def get_evaluator(self):
        from dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator
        from config.maze2d_constraints import con_groups
        # Raise the recursion limit to avoid problems when parsing formulas
        sys.setrecursionlimit(10000)
        # Setting TL_RECORD_TRACE asks DTL to record the evaluation trace.
        # Using this we can find the conflicting part between logits and formula.
        os.environ['TL_RECORD_TRACE'] = '1'

        evaluator = DTL_Cont_Cons_Evaluator(device='cuda')
        evaluator.set_atomic_props(con_groups[self.dataset.env.name])
        return evaluator

    def option2trj(self, option):
        option_v = torch.zeros_like(option)
        option_obs = torch.cat([option, option_v], dim=2)
        states = option_obs[:,-1,:]
        placeholder_action = torch.zeros_like(option)
        trj = torch.cat([placeholder_action, option_obs], dim=2)
        ## [ n_samples x horizon-1 x observation_dim ]
        normed_observations = to_np(option_obs)
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
        return trj, observations, states
