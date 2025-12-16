from collections import namedtuple
import numpy as np
import torch
import einops
import pdb

from diffuser.utils import to_np, to_torch, apply_dict

Trajectories = namedtuple('Trajectories', 'actions observations')


class Policy:

    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):

        conditions = self._format_conditions(conditions, batch_size)
        ## run reverse diffusion process
        sample, chains = self.diffusion_model(conditions, return_diffusion=True)
        sample = to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        if 0 != self.action_dim:
            actions = self.normalizer.unnormalize(actions, 'actions')
        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        chains = self.normalizer.unnormalize(to_np(chains[:,:,:,self.action_dim:]), 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories, chains, sample


class CondPolicy(Policy):

    def __init__(self, cond_dim, **kwargs):
        super().__init__(**kwargs)
        self.cond_dim = cond_dim  # the dimension of conditioned start state

    def _extract_state(self, obs):
        return obs[:self.cond_dim]

    def _format_conditions(self, conditions, batch_size):
        conditions = apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = apply_dict(
            self._extract_state,
            conditions,
        )
        conditions = apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions


class QPolicy(Policy):

    def __init__(self, get_option, q, diffusion_model, normalizer, cond_dim, n_o, guide, **sample_kwargs):
        super().__init__(diffusion_model, normalizer)

        self.q = q
        self.get_option = get_option
        self.cond_dim = cond_dim
        self.n_o = n_o
        self.guide = guide
        self.sample_kwargs = sample_kwargs

    def update_ltl_graph(self, ltl_tup, ltls_all_tup, graphs_all):
        idx_ltl_all_tup = ltls_all_tup.index(ltl_tup)
        ltl_graph = graphs_all[idx_ltl_all_tup]
        self.ltls_graph = ltl_graph[None,:]

    def __call__(self, conditions, batch_size=1, has_action=True):
        conditions = self._format_conditions(conditions, batch_size)
        states = conditions[0]

        option, option_obs = self.get_option(q = self.q,
                                             s = states,
                                             ltl = self.ltls_graph,
                                             diffusion = self.diffusion_model,
                                             cond_dim = self.cond_dim,
                                             n_o = self.n_o,
                                             guide = self.guide,
                                             **self.sample_kwargs)
        placeholder_actions = torch.zeros_like(option)
        sample = torch.cat([placeholder_actions, option_obs], dim=2)
        sample = to_np(sample)

        ## [ batch_size x horizon-1 x observation_dim ]
        normed_observations = to_np(option_obs)
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        action = None
        chains = None

        trajectories = Trajectories(to_np(placeholder_actions), observations)
        return action, trajectories, chains, sample
