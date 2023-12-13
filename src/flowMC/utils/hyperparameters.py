flowmc_default_hyperparameters = {
    "n_loop_training": 3,
    "n_loop_production": 3,
    "n_loop_pretraining": 0,
    "n_local_steps": 50,
    "n_global_steps": 50,
    "n_chains": 20,
    "n_epochs": 30,
    "learning_rate": 0.001,
    "max_samples": 100000,
    "momentum": 0.9,
    "batch_size": 10000,
    "use_global": True,
    "logging": True,
    "keep_quantile": 0,
    "local_autotune": None,
    "train_thinning": 1,
    "output_thinning": 1,
    "n_sample_max": 10000,
    "precompile": False,
    "verbose": False,
    "outdir_name": "./outdir/"
}

flowmc_hyperparameters_explanation = {
    "n_loop_training": "(int): Number of training loops.",
    "n_loop_production": "(int): Number of production loops.",
    "n_loop_pretraining": "(int): Number of pretraining loops.",
    "n_local_steps": "(int) Number of local steps per loop.",
    "n_global_steps": "(int) Number of local steps per loop.",
    "n_chains": "(int) Number of chains",
    "n_epochs": "(int) Number of epochs to train the NF per training loop",
    "learning_rate": "(float) Learning rate used in the training of the NF",
    "max_samples": "(int) Maximum number of samples fed to training the NF model",
    "momentum": "(float) Momentum used in the training of the NF model with the Adam optimizer",
    "batch_size": "(int) Size of batches used to train the NF",
    "use_global": "(bool) Whether to use an NF proposal as global sampler",
    "logging": "(bool) Whether to log the training process",
    "keep_quantile": "Quantile of chains to keep when training the normalizing flow model",
    "local_autotune": "(Callable) Auto-tune function for the local sampler",
    "train_thinning": "(int) Thinning parameter for the data used to train the normalizing flow",
    "output_thinning": "(int) Thinning parameter with which to save the data ",
    # TODO is this n_sample_max duplicate with max_samples above? Or is this something else?
    "n_sample_max": "(int) Maximum number of samples fed to training the NF model",
    "precompile": "(bool) Whether to precompile",
    "verbose": "(bool) Show steps of algorithm in detail",
    "outdir_name": "(str) Location to which to save plots, samples and hyperparameter settings. Note: should ideally start with `./` and also end with `/`"
}

### TODO if not used, then delete
# flowmc_hyperparameters_names = list(flowmc_default_hyperparameters.keys())
# def update_hyperparameters(dict1: dict, dict2: dict):
#     """Update dict1 with values of dict2 but only consider the keys given."""
#     dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())
#     return dict1
