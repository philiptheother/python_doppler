import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T')
]

rl_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('discount', '\u03b3'),
]

test_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('discount', '\u03b3'),
]


base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'dim_mults': (1, 4, 8),
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'loss_type': 'l2',
        'clip_denoised': True,

        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.LocationDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_no_goal'],
        'use_padding': False,
        'max_n_episodes': 1,  # the whole trajectory in the dataset
        'max_path_length': 5_000_000,  # the whole trajectory in the dataset

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),
        'device': 'cuda',

        ## training
        'n_steps_per_epoch': 10000,
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,

        ## planning
        'batch_size_plan': 1,
        'num_avg': 5,
        'vis_freq': 10,
    },

    'rl': {
        ## model
        'model': 'models.RGCN_MLP_LN',
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.RLRewardDataset',
        'horizon': 32,
        'discount': 0.997,
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_no_goal'],
        'use_padding': False,
        'max_n_episodes': 1,  # the whole trajectory in the dataset
        'max_path_length': 5_000_000,  # the whole trajectory in the dataset

        ## serialization
        'logbase': 'logs',
        'prefix': 'rl/',
        'exp_name': watch(rl_args_to_watch),
        'device': 'cuda',

        ## training
        'trainer': 'utils.RLTrainer',
        'diffusion_loadpath': None,  # override this in specific experiments down below
        # RL training hyper-parameters
        'n_steps_per_epoch': 10000,
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'a_n_n_target': 16,
        'a_n_noise': True,
        'policy_noise': 0.02,
        'noise_clip': 0.05,

        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'n_reference': 50,

        ## planning
        'n_episodes': 20,
    },

    'test': {
        'batch_size': 1,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,

        ## receding horizon control
        'max_n_episode': 50,

        ## serialization
        'logbase': 'logs',
        'prefix': 'tests/release',
        'exp_name': watch(test_args_to_watch),
        'device': 'cuda',

        ## loading
        'q_value_loadpath': 'f:orl/H{horizon}_γ0.997/',
        'q_value_epoch': 'latest',

        ## guide sample_kwargs
        'n_guide_steps': {'default':2},
        'scale': {'default':1},
        't_stopgrad': {'default':2},
        'scale_grad_by_std': {'default':True},

        'planning_only': False
    },
}

#------------------------ overrides ------------------------#

maze2d_medium_v1 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 16,

        'n_steps_per_epoch': 500,
        'n_train_steps': 125_000,
        'batch_size': 512,

        'sample_freq': 200,
        'n_samples': 9,
    },
    'rl':{
        'horizon': 32,

        'diffusion_loadpath': 'f:diffusion/H{horizon}_T16/cuda118',

        'n_steps_per_epoch': 500,
        'n_train_steps': 125_000,
        'batch_size': 512,

        'sample_freq': 200,
    },
    'test': {
        'horizon': 32,
        'n_diffusion_steps': 16,

        'q_value_loadpath': 'f:rl/H{horizon}_γ0.997/cuda118',

        'n_guide_steps': {'max_q':1,},
        'scale': {'max_q':1,},
        't_stopgrad': {'max_q':2,},
        'scale_grad_by_std': {'max_q':False,},

        'planning_only': False
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 16,

        'n_steps_per_epoch': 500,
        'n_train_steps': 125_000,
        'batch_size': 512,

        'sample_freq': 200,
        'n_samples': 9,
    },
    'rl':{
        'horizon': 32,

        'diffusion_loadpath': 'f:diffusion/H{horizon}_T16/cuda118',

        'n_steps_per_epoch': 500,
        'n_train_steps': 125_000,
        'batch_size': 512,

        'sample_freq': 200,
    },
    'test': {
        'horizon': 32,
        'n_diffusion_steps': 16,

        'q_value_loadpath': 'f:rl/H{horizon}_γ0.997/cuda118',

        'n_guide_steps': {'max_q':1,},
        'scale': {'max_q':1,},
        't_stopgrad': {'max_q':2,},
        'scale_grad_by_std': {'max_q':False,},

        'planning_only': False
    },
}
