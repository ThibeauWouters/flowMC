hyperparameters = dict()

# TODO improve organisation here

# n_loop_training (int, optional): Number of training loops. Defaults to 3.
# n_loop_production (int, optional): Number of production loops. Defaults to 3.
# n_local_steps (int, optional): Number of local steps per loop. Defaults to 50.
# n_global_steps (int, optional): Number of global steps per loop. Defaults to 50.
# n_chains (int, optional): Number of chains. Defaults to 20.
# n_epochs (int, optional): Number of epochs per training loop. Defaults to 30.
# learning_rate (float, optional): Learning rate for the NF model. Defaults to 0.01.
# max_samples (int, optional): Maximum number of samples fed to training the NF model. Defaults to 10000.
# momentum (float, optional): Momentum for the NF model. Defaults to 0.9.
# batch_size (int, optional): Batch size for the NF model. Defaults to 10000.
# use_global (bool, optional): Whether to use global sampler. Defaults to True.
# logging (bool, optional): Whether to log the training process. Defaults to True.
# keep_quantile (float, optional): Quantile of chains to keep when training the normalizing flow model. Defaults to 0..
# local_autotune (None, optional): Auto-tune function for the local sampler. Defaults to None.
# train_thinning (int, optional): Thinning for the data used to train the normalizing flow. Defaults to 1.

hyperparameters = {
    "n_loop_training": 3,
    "n_loop_production": 3,
    "n_loop_pretraining": 3,
    "n_local_steps": 50,
    "n_global_steps": 50,
    "n_chains": 20,
    "n_epochs": 30,
    "learning_rate": 0.01,
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

hyperparameters_keys = list(hyperparameters.keys())

def update_hyperparameters(dict1: dict, dict2: dict):
    """Update dict1 with values of dict2 but only consider the keys given."""
    dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())
    return dict1
