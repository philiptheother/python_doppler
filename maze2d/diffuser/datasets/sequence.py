import os
import sys
import json
from collections import namedtuple
import random
import numpy as np
import sparse
import torch
from tqdm import tqdm
import gym

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
RLBatch = namedtuple('RLBatch', 'trajectories states options rewards_cumulative states_next terminal index_trj index_ltl')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, 
        output=print, **kwargs):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn, output=output, **kwargs)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize(output)

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        output(fields)
        self.output = output
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class LocationDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition only on current locations for planning
        '''
        return {0: observations[0,:2]}

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class RLRewardDataset(LocationDataset):

    def __init__(self,
                 *args,
                 discount,
                 props=['d0','d1','d2','d3','d4','d5'],
                 filename_ltls='datasets/ltls_until.txt',
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.props = props
        from diffuser.utils.ltl_parser import LTLParser
        ltl_str2tup_converter = LTLParser(propositions=self.props)

        # load all goal ltls
        with open(filename_ltls) as file:
            ltls_goal_str=file.read()
            ltls_goal_str=ltls_goal_str.replace('p','d')  # use 'd' instead of 'p' to make spot work
            ltls_goal_str=ltls_goal_str.split("\n")[0:-1]
        self.ltls_goal_str = ltls_goal_str
        self.ltls_goal_tup = [ltl_str2tup_converter(ltl_goal_str) for ltl_goal_str in self.ltls_goal_str]
        
        self.known_progressions = {}
        ltls_all_tup = self.progression_all(self.ltls_goal_tup)
        self.ltls_all_tup = ltls_all_tup[1:]
        assert len(self.ltls_all_tup) == len(set(self.ltls_all_tup))

        self.n_ltl = len(self.ltls_all_tup)
        self.n_train_ltls = ltls_all_tup[0]
        self.n_test_ltls = self.n_ltl - self.n_train_ltls

        self.graphs = self._get_graphs(self.ltls_all_tup)
        
        self.name_value = 'rl'
        filename_r = 'datasets/' + '-'.join(self.env.name.split('-')[:-1] + ["H"+str(self.horizon), self.name_value, 'rewards.npz'])
        filename_d = 'datasets/' + '-'.join(self.env.name.split('-')[:-1] + ["H"+str(self.horizon), self.name_value, 'dones.npz'])
        filename_idx_ltl_next = 'datasets/' + '-'.join(self.env.name.split('-')[:-1] + ["H"+str(self.horizon), self.name_value, 'idx_ltl_next.npy'])

        if os.path.isfile(filename_r) and os.path.isfile(filename_d) and os.path.isfile(filename_idx_ltl_next):
            self.rewards = sparse.load_npz(filename_r)
            self.dones = sparse.load_npz(filename_d)
            self.idx_ltl_next = np.load(filename_idx_ltl_next)
            assert len(self.rewards) == super().__len__()
            assert len(self.dones) == super().__len__()
            assert len(self.idx_ltl_next) == super().__len__()

            self.output(f'[ datasets/sequence ] Load generated rewards, dones and progressed ltl of {self.name_value} formulation from:\n{filename_r},\n{filename_d},\n{filename_idx_ltl_next}')
        else:
            self.rewards, self.dones, self.idx_ltl_next = self._get_values()
            assert len(self.rewards) == super().__len__()
            assert len(self.dones) == super().__len__()
            assert len(self.idx_ltl_next) == super().__len__()

            sparse.save_npz(filename_r, self.rewards)
            sparse.save_npz(filename_d, self.dones)
            np.save(filename_idx_ltl_next, self.idx_ltl_next)

            self.output(f'[ datasets/sequence ] Save generated rewards, dones and progressed ltl of {self.name_value} formulation to:\n{filename_r},\n{filename_d},\n{filename_idx_ltl_next}')

        self.state_dim = self.get_state_dim()
        self.option_dim = self.state_dim*(self.horizon - 1)  # future waypoints
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.horizon-1)  # for receding horizon only where option_horizon is the option timesteps

    def get_state_dim(self):
        if hasattr(self, 'state_dim'):
            return self.state_dim
        else:
            self.state_dim = self.observation_dim - 2  # dimensionality of the states used for Q function
            return self.state_dim

    def progression(self, ltl_formula, truth_assignment):
        from diffuser.utils.ltl_progression import progress_and_clean

        if (ltl_formula, truth_assignment) not in self.known_progressions:
            result_ltl = progress_and_clean(ltl_formula, truth_assignment)
            self.known_progressions[(ltl_formula, truth_assignment)] = result_ltl

        return self.known_progressions[(ltl_formula, truth_assignment)]

    def progression_dfs(self, ltl, ltl_all):
        if ltl not in ltl_all:
            ltl_all.append(ltl)
            if 'True' != ltl and 'False' != ltl:
                for prop in self.props:
                    ltl_prog = self.progression(ltl, prop)
                    if ltl_prog != ltl and ltl_prog not in ltl_all:
                        self.progression_dfs(ltl_prog, ltl_all)

    def progression_all(self, ltls_goal):
        # The first line of ltls_all contains the total number of progressed training ltls
        n_ltls_goal = len(ltls_goal)
        n_ltls_goal_train = int(0.8*n_ltls_goal)
        
        ltls_all = []
        for idx in tqdm(range(n_ltls_goal_train)):
            ltl_goal = ltls_goal[idx]
            self.progression_dfs(ltl_goal, ltls_all)

        ltls_all.insert(0, len(ltls_all))

        for idx in tqdm(range(n_ltls_goal_train,n_ltls_goal)):
            ltl_goal = ltls_goal[idx]
            self.progression_dfs(ltl_goal, ltls_all)

        return ltls_all

    def _get_graphs(self, ltls_tup):
        from diffuser.utils.ast_builder import ASTBuilder
        tree_builder = ASTBuilder(propositions=self.props)
        graphs = np.array([[tree_builder(ltl_tup).to('cuda')] for ltl_tup in ltls_tup])

        ltl_embed_output_dim = 32
        for i in range(graphs.shape[0]):
            # tmp = graphs[i][0].nodes[None].data['is_root'].detach()
            d = graphs[i][0].nodes[None].data['feat'].size()
            root_weight = torch.ones((1, ltl_embed_output_dim))
            others_weight = torch.zeros((d[0]-1, ltl_embed_output_dim))
            weight = torch.cat([root_weight, others_weight])
            graphs[i][0].nodes[None].data['is_root'] = weight.cuda()
            # print(f"graph nodes = {graphs[i][0].nodes[None].data}")
        return graphs

    def boundary2prop(self, assignment):
        """
            calculate the assigned proposition for an assignment on every low level constraints

            Input
                assignment: numpy.array (num_constraints, time_steps)

            Return
                propositions: numpy.array of str [i]
        """
        assert 0 == assignment.shape[0] % 4
        n = assignment.shape[0] // 4
        t = assignment.shape[1]
        # Reshape the input array A to shape (n, 4, t)
        assignment_reshaped = assignment.reshape(n, 4, -1)
        # Check if all elements in each group of 4 elements are positive
        all_positive = np.all(assignment_reshaped > 0, axis=1).astype(int)
        mask = np.any(all_positive, axis=0)  # True if there is event at t
        strings_list = []
        for col in range(t):
            if mask[col]:
                # Find the index of the 1 in the current column
                index = np.argmax(all_positive[:, col])
                # Append the corresponding string to the list
                strings_list.append(f'd{index}')
            else:
                strings_list.append("")
        return strings_list

    def _get_assignments(self):
        filename_assigns = 'datasets/' + '-'.join(self.env.name.split('-')[:-1] + ["H"+str(self.horizon), self.name_value, 'assignments.json'])
        n_episodes = len(self.fields.path_lengths)

        if os.path.isfile(filename_assigns):
            with open(filename_assigns, 'r') as fjson:
                assignments = json.load(fjson)  # loaded type is list
            self.output(f'[ datasets/sequence ] Load generated assignments of constraints from {filename_assigns}')
            assert len(assignments) == n_episodes
            for i in range(n_episodes):
                assert len(assignments[i]) == self.fields.path_lengths[i]

        else:
            self.output('[ datasets/sequence ] Getting constraints satisfaction assignments for all items...')

            from dtl.dtl_cont_cons import DTL_Cont_Cons_Evaluator
            from config.maze2d_constraints import con_groups
            # Raise the recursion limit to avoid problems when parsing formulas
            sys.setrecursionlimit(10000)
            # Setting TL_RECORD_TRACE asks DTL to record the evaluation trace.
            # Using this we can find the conflicting part between logits and formula.
            os.environ['TL_RECORD_TRACE'] = '1'

            evaluator = DTL_Cont_Cons_Evaluator(device='cuda')
            evaluator.set_atomic_props(con_groups[self.env.name])

            assignments = []
            batch_size = 512
            for i in range(n_episodes):
                assignments.append([])
                for j in tqdm(range(0, self.fields.path_lengths[i], batch_size)):
                    j_end = min(self.fields.path_lengths[i], j+batch_size)
                    observation = self.fields.normed_observations[i, j:j_end]
                    action = self.fields.normed_actions[i, j:j_end]
                    trj = np.concatenate([action, observation], axis=-1)
                    trj = torch.tensor(trj, dtype=torch.float32).cuda()
                    if trj.ndim < 2:
                        trj = trj.unsqueeze(0).unsqueeze(0)
                    else:
                        trj = trj.unsqueeze(0)
                    assignment = evaluator.get_assignments(trj)
                    assignment = self.boundary2prop(assignment[0].cpu().numpy())
                    assignments[i] += assignment

            assert len(assignments) == n_episodes
            for i in range(n_episodes):
                assert len(assignments[i]) == self.fields.path_lengths[i]

            with open(filename_assigns, 'w') as fjson:
                json.dump(assignments, fjson)  # saved type is list
            self.output(f'[ datasets/sequence ] Save generated assignments of proposition to {filename_assigns}')

        return assignments

    def _get_values(self):
        '''
            Use LTL progression to label the entire cumulative rewards of a segment since the goal ltl might change inside a given segment, and store the resulting progressed LTL
        '''
        self.output('[ datasets/sequence ] Getting LTL reward values for all items...')
        assignments = self._get_assignments()

        rewards_coords = []
        rewards_data = []
        dones_coords = []
        dones_data = []
        idx_ltl_next = np.zeros([super().__len__(), self.n_ltl])
        # ltl_next = [[None]*self.n_ltl]*super().__len__()

        for idx_ltl in tqdm(range(self.n_ltl)):
            ltl_goal_tup = self.ltls_all_tup[idx_ltl]
            self.output(f'[ datasets/sequence ] getting values for {idx_ltl}-th LTL: {ltl_goal_tup}')

            for idx_trj in tqdm(range(super().__len__())):
                ltl_tup = ltl_goal_tup

                path_ind, start, end = self.indices[idx_trj]
                assignments_trj = assignments[path_ind][start:end]
                # ignore progressing the ltl with current location
                # ltl_tup = self.progression(ltl_tup, assignments_trj[0])

                # Loop over each possible sub trj length h=2,3,...,H
                for h in range(1, self.horizon):
                    ltl_tup = self.progression(ltl_tup, assignments_trj[h])
                    # Computing the LTL reward and done signal
                    ltl_reward = 0
                    ltl_done   = 0
                    if ltl_tup == 'True':
                        ltl_reward = 1
                        ltl_done   = 1
                    elif ltl_tup == 'False':
                        ltl_reward = -1
                        ltl_done   = 1

                    if 1 == ltl_done:
                        # Store the output values in the corresponding column of the output matrix
                        rewards_coords.append((idx_trj, h-1, idx_ltl))
                        rewards_data.append(ltl_reward)
                        dones_coords.append((idx_trj, idx_ltl))
                        dones_data.append(ltl_done)
                        break
                idx_ltl_next[idx_trj][idx_ltl] = self.ltls_all_tup.index(ltl_tup)
                # ltl_next[idx_trj][idx_ltl] = ltl_tup
            # end of all ltls
        # Create the COO array from the coordinates and data
        rewards_coords = np.array(rewards_coords).T  # Transpose to match COO format
        dones_coords = np.array(dones_coords).T  # Transpose to match COO format
        rewards = sparse.COO(rewards_coords, rewards_data, shape=(super().__len__(), self.horizon-1, self.n_ltl))
        dones = sparse.COO(dones_coords, dones_data, shape=(super().__len__(), self.n_ltl))
        return rewards, dones, idx_ltl_next

    def __getitem__(self, idx, eps=1e-4):
        '''
            Get state, goal ltl, option, reward, next state and next ltl of an ltl task

            Require:
                Calculated Non-Markov reward (penalty) for every location
        '''
        idx_trj = idx
        idx_ltl = random.randint(0, self.n_train_ltls-1)

        path_ind, start, end = self.indices[idx_trj]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        trajectories = np.concatenate([actions, observations], axis=-1)

        states = observations[0, :self.state_dim]
        options = observations[1:, :self.state_dim]
        states_next = observations[-1, :self.state_dim]

        rewards = self.rewards[idx_trj, :self.horizon-1, idx_ltl]

        rewards_cumulative = (self.discounts * rewards).sum()
        rewards_cumulative = np.array([rewards_cumulative], dtype=np.float32)

        terminal = np.array([self.dones[idx_trj,idx_ltl]], dtype=np.float32)

        idx_trj = np.array(idx_trj, dtype=int)
        idx_ltl = np.array(idx_ltl, dtype=int)
    
        batch = RLBatch(trajectories, states, options, rewards_cumulative, states_next, terminal, idx_trj, idx_ltl)
        return batch
