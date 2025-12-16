import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('rl')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath        = (args.savepath, 'dataset_config.pkl'),
    output          = args.logger.info,
    # _class parameters
    env             = args.dataset,
    horizon         = args.horizon,
    discount        = args.discount,
    normalizer      = args.normalizer,
    preprocess_fns  = args.preprocess_fns,
    use_padding     = args.use_padding,
    max_n_episodes  = args.max_n_episodes,
    max_path_length = args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    output=args.logger.info,
    # _class parameters
    env=args.dataset,
)

dataset = dataset_config(output=args.logger.info)
renderer = render_config(output=args.logger.info)

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    output=args.logger.info,
    # _class parameters
    state_dim = dataset.state_dim,
    num_inputs = dataset.state_dim + dataset.option_dim,
    device=args.device,
)

trainer_config = utils.Config(
    args.trainer if hasattr(args, 'trainer') else utils.Trainer,
    savepath            = (args.savepath, 'trainer_config.pkl'),
    output              = args.logger.info,
    # _class parameters
    option_horizon      = args.horizon-1,
    discount            = args.discount,
    model_config        = model_config,
    diffusion_loadpath  = args.diffusion_loadpath,
    train_batch_size    = args.batch_size,
    train_lr            = args.learning_rate,
    a_n_n_target        = args.a_n_n_target,
    policy_noise        = args.policy_noise,
    noise_clip          = args.noise_clip,

    sample_freq         = args.sample_freq,
    save_freq           = args.save_freq,
    label_freq          = int(args.n_train_steps // args.n_saves),
    results_folder      = args.savepath,
    n_reference         = args.n_reference,
    n_episodes          = args.n_episodes,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

trainer = trainer_config(dataset, renderer, output=args.logger.info)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(trainer.network, topk=5)

args.logger.info('Testing forward...')
batch = utils.batchify(dataset[0])
loss = trainer.batch_rl(*batch)
args.logger.info('✓')
args.logger.info('Testing backward...')
loss.backward()
args.logger.info('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    args.logger.info(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)
