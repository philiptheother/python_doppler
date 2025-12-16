import pdb
from tqdm import tqdm

import os
import sys
import json
from os.path import join

import numpy as np
import torch

import diffuser.guides as guides
import diffuser.datasets as datasets
import diffuser.utils as utils

from dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator
sys.setrecursionlimit(10000)
os.environ['TL_RECORD_TRACE'] = '1'

from config.maze2d_constraints import con_groups

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('test')

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

Q_value_experiment = utils.load_q(args.logbase, args.dataset, args.q_value_loadpath, epoch=args.q_value_epoch)
dataset = Q_value_experiment.dataset
renderer = Q_value_experiment.renderer
trainer = Q_value_experiment.trainer
diffusion = trainer.diffusion
q_network = trainer.network

guide_config = utils.Config('guides.PSDPPGuide',
                            state_dim = dataset.state_dim,
                            action_dim = dataset.action_dim,
                            verbose = False)
guide_dpp = guide_config()

policies = {
    'max_q':  guides.QPolicy(q                  = q_network,
                             get_option         = trainer.get_c_max_o_diffusion,
                             diffusion_model    = diffusion,
                             normalizer         = dataset.normalizer,
                             cond_dim           = trainer.cond_dim,
                             n_o                = 4,
                             guide              = guide_dpp,
                             # sampling kwargs
                             sample_fn          = guides.n_step_fps_p_sample,
                             n_guide_steps      = args.n_guide_steps.get('max_q'),
                             scale              = args.scale.get('max_q'),
                             t_stopgrad         = args.t_stopgrad.get('max_q'),
                             scale_grad_by_std  = args.scale_grad_by_std.get('max_q'),
                             return_diffusion   = True,
                             return_obs         = True,),
}

#---------------------------------- tester ----------------------------------#

props = dataset.props
ltls_goal_str = dataset.ltls_goal_str
ltls_goal_tup = dataset.ltls_goal_tup
ltls_all_tup = dataset.ltls_all_tup
graphs_all = dataset.graphs

n_ltl = len(ltls_goal_str)
n_train_ltls = int(0.8*n_ltl)

tester_dtl = DTL_Cont_Cons_Evaluator(device='cuda')
tester_dtl.set_atomic_props(con_groups[args.dataset])

#---------------------------------- main loop ----------------------------------#

n_restart = 10

metrics = ['success_rollout','failure_rollout','steps']
results = {}
for key, val in policies.items():
    results[key] = {}
    for metric in metrics:
        results[key][metric] = {'all': np.zeros((n_ltl,n_restart)).tolist(),
                                'mean': np.zeros(n_ltl).tolist(),
                                'avg': 0.0, 'avg_train': 0.0, 'avg_test': 0.0,}

for idx in tqdm(range(n_ltl)):
    ltl_goal_str = ltls_goal_str[idx]
    tester_dtl.set_ltl_formula(ltl_goal_str)
    ltl_goal_tup = ltls_goal_tup[idx]
    args.logger.info('[ scripts/test_maze2d ] Set {}-th LTL as: {}'.format(idx, ltl_goal_str))

    for seed in range(n_restart):
        state_start = env.reset()

        for key, policy in policies.items():
            env.set_state(state_start[0:2], state_start[2:4])

            plan = []
            rollout_obs = []
            rollout_actions = []

            state = env.state_vector().copy()
            ltl_tup = ltl_goal_tup
            steps = 0
            for e in range(args.max_n_episode):

                rollout_obs_horizon = []
                rollout_actions_horizon = []

                if args.planning_only:
                    cond = {0: state,}
                    policy.update_ltl_graph(ltl_tup, ltls_all_tup, graphs_all)
                    _, samples, _, sample = policy(cond, batch_size=args.batch_size)

                    plan.append(samples.observations)
                    observations = dataset.normalizer.normalize(samples.observations, 'observations')
                    placeholder_action = samples.actions
                    trj = torch.cat([torch.tensor(placeholder_action), torch.tensor(observations)], dim=2)

                    state = samples.observations[0,-1,:]

                else:
                    for t in range(args.horizon-1):
                        state = env.state_vector().copy()
                        if t == 0:
                            cond = {0: state,}
                            policy.update_ltl_graph(ltl_tup, ltls_all_tup, graphs_all)
                            _, samples, _, sample = policy(cond, batch_size=args.batch_size)
                            # actions = samples.actions[0]
                            sequence = samples.observations[0]
                            plan.append(sequence)

                        # next location in path, copy the last location at max timestep of sequence onwards
                        if t < len(sequence) - 1:
                            next_waypoint = sequence[t+1]
                        else:
                            next_waypoint = sequence[-1].copy()
                            next_waypoint[2:] = 0

                        ## can use actions or define a simple controller based on state predictions
                        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

                        next_observation, reward, terminal, _ = env.step(action)

                        ## update rollout observations
                        rollout_obs.append(next_observation.copy())
                        rollout_actions.append(action.copy())

                        rollout_obs_horizon.append(next_observation.copy())
                        rollout_actions_horizon.append(action.copy())

                        if terminal:
                            break

                    rollout_obs_horizon = dataset.normalizer.normalize(np.array(rollout_obs_horizon), 'observations')
                    rollout_actions_horizon = dataset.normalizer.normalize(np.array(rollout_actions_horizon), 'actions')
                    rollout_obs_horizon = np.array(rollout_obs_horizon)[None,:,:]
                    rollout_actions_horizon = np.array(rollout_actions_horizon)[None,:,:]
                    trj = torch.cat([torch.tensor(rollout_actions_horizon), torch.tensor(rollout_obs_horizon)], dim=2)

                assignments = tester_dtl.get_assignments(trj)
                assignments = dataset.boundary2prop(assignments[0].cpu().numpy())
                for a in assignments:
                    ltl_tup = dataset.progression(ltl_tup, a)
                    steps += 1
                    # Computing the LTL reward and done signal
                    ltl_done   = False
                    if ltl_tup == 'True':
                        ltl_done   = True
                    elif ltl_tup == 'False':
                        ltl_done   = True

                    if ltl_done:
                        break
                if ltl_done:
                    break

            savepath = join(args.savepath, 'sample', f'ltl{idx}_seed{seed}_{key}_plan_rollout.png')
            if args.planning_only:
                renderer.composite(savepath, np.concatenate(plan), ncol=1)
                np.save('.'.join(savepath.split('.')[:-1] + ['npy']), {'sample':np.concatenate(plan, axis=0),})
            else:
                renderer.composite(savepath, (np.concatenate(plan, axis=0), np.array(rollout_obs)), ncol=2)
                np.save('.'.join(savepath.split('.')[:-1] + ['npy']), {'sample':np.concatenate(plan, axis=0),
                                                                       'rollout':np.array(rollout_obs)})
            # end of rollout
            results[key]['success_rollout']['all'][idx][seed] = 1.0 if ltl_tup == 'True' else 0.0
            results[key]['failure_rollout']['all'][idx][seed] = 1.0 if ltl_tup == 'False' else 0.0
            results[key]['steps']['all'][idx][seed] = steps
        # end of all methods
        for _method, _ in results.items():
            s = _method + ":\t"
            for _metric in results[_method].keys():
                s += _metric + "(" + str(results[_method][_metric]['all'][idx][seed]) + ")\t"
            args.logger.info(s)
    # end of all seeds
    for _method, _ in results.items():
        s = _method + " mean:\t"
        for _metric in results[_method].keys():
            _mean = np.mean(results[_method][_metric]['all'][idx])
            results[_method][_metric]['mean'][idx] = _mean
            s += _metric + "(" + str(_mean) + ")\t"
        args.logger.info(s)
# end of all ltls
for _method, _ in results.items():
    s = _method + " avg:\t"
    for _metric in results[_method].keys():
        _avg = np.mean(results[_method][_metric]['mean'])
        _avg_train = np.mean(results[_method][_metric]['mean'][0:n_train_ltls])
        _avg_test = np.mean(results[_method][_metric]['mean'][n_train_ltls:n_ltl])
        results[_method][_metric]['avg'] = _avg
        results[_method][_metric]['avg_train'] = _avg_train
        results[_method][_metric]['avg_test'] = _avg_test
        s += _metric + "(" + str(_avg) + ")" + "(" + str(_avg_train) + ")" + "(" + str(_avg_test) + ")\t"
    args.logger.info(s)

# average successful and non-failure steps
for _method, _ in results.items():
    # s = _method + " mean_succ:\t"
    results[_method]['steps']['mean_succ'] = np.zeros(n_ltl).tolist()
    results[_method]['steps']['mean_nonfail'] = np.zeros(n_ltl).tolist()
    _metric = 'steps'

    for idx in range(n_ltl):
        _value = np.array(results[_method][_metric]['all'][idx])
        _mask_succ = np.array(results[_method]['success_rollout']['all'][idx])
        _mask_nonfail = 1.0 - np.array(results[_method]['failure_rollout']['all'][idx])
        _mean_succ = np.sum(_value*_mask_succ)/np.sum(_mask_succ)
        _mean_nonfail = np.sum(_value*_mask_nonfail)/np.sum(_mask_nonfail)
        results[_method][_metric]['mean_succ'][idx] = _mean_succ
        results[_method][_metric]['mean_nonfail'][idx] = _mean_nonfail
        # args.logger.info(s + _metric + "(" + str(_mean_succ) + ")\t")

    s = _method + " avg_succ:\t"
    _avg_succ = np.mean(results[_method][_metric]['mean_succ'])
    _avg_succ_train = np.mean(results[_method][_metric]['mean_succ'][0:n_train_ltls])
    _avg_succ_test = np.mean(results[_method][_metric]['mean_succ'][n_train_ltls:n_ltl])
    results[_method][_metric]['avg_succ'] = _avg_succ
    results[_method][_metric]['avg_succ_train'] = _avg_succ_train
    results[_method][_metric]['avg_succ_test'] = _avg_succ_test
    s += _metric + "(" + str(_avg_succ) + ")" + "(" + str(_avg_succ_train) + ")" + "(" + str(_avg_succ_test) + ")\t"
    args.logger.info(s)

    s = _method + " avg_nonfail:\t"
    _avg_nonfail = np.mean(results[_method][_metric]['mean_nonfail'])
    _avg_nonfail_train = np.mean(results[_method][_metric]['mean_nonfail'][0:n_train_ltls])
    _avg_nonfail_test = np.mean(results[_method][_metric]['mean_nonfail'][n_train_ltls:n_ltl])
    results[_method][_metric]['avg_nonfail'] = _avg_nonfail
    results[_method][_metric]['avg_nonfail_train'] = _avg_nonfail_train
    results[_method][_metric]['avg_nonfail_test'] = _avg_nonfail_test
    s += _metric + "(" + str(_avg_nonfail) + ")" + "(" + str(_avg_nonfail_train) + ")" + "(" + str(_avg_nonfail_test) + ")\t"
    args.logger.info(s)

## save results as a json file
json_path = join(args.savepath, 'results.json')
json.dump(results, open(json_path, 'w'), indent=2, sort_keys=True)
