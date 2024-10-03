opt_data = {
    "frame_rate": 30,
    "input_length": 60,
    "input_mode": "bones",
    # normalise data to zero mean unit var
    "normalise": True,
    "output_length": 30,
    # during training generate a sample from every x th frame
    "skip_rate": 90,
    # filter unrealistic Kinect movement during training
    "threshold_filter": 0.4,
    # train the network on sequences with "static" ground truth
    "use_idle": False,
}

opt_rnn = {
    "dropout": 0,
    "init_glorot": False,
    "lstm": True,
    "num_hidden": 1024,
    "num_layers": 1,
}

opt_transformer = {
    "dropout": 0.1,
    "num_embedding": 128,
    "num_heads": 8,
    "num_hidden": 256,
    "num_layers": 8,
    "person_attn": False,
    "scale_input": False,
    "temporal_mask": False,
    # use the learning rate schedule of the original transformer paper
    # True will override other lr schedules
    "transformer_lr": False,
}

opt_training = {
    "batch_size": 32,
    "betas": (0.9, 0.98),
    "clip_grad_norm": 5,
    "importance_abs": 0.5,
    # increase output length every x epochs
    # 0 means that network always predicts output_length frames
    "inc_len_rate": 3,
    "learning_rate": 0.001,
    "loss": "l1",
    "lr_exp_decay": 0.98,
    # additional lr decay by factor 10 at epochs
    "milestones": [120],
    "num_epochs": 150,
    "weight_decay": 0,
}
