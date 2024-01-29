import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner
from typing import Callable, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Int, Float
import optax
import equinox as eqx

import jax

from flowMC.nfmodel.utils import make_training_loop
from flowMC.sampler.NF_proposal import NFProposal
from flowMC.sampler.Proposal_Base import ProposalBase
from flowMC.nfmodel.base import NFModel
from flowMC.utils import gelman_rubin, get_mean_and_std_chains, initialize_summary_dict, default_corner_kwargs, IMRPhenomD_labels
from flowMC.utils.hyperparameters import flowmc_default_hyperparameters

class Sampler:
    """
    Sampler class that host configuration parameters, NF model, and local sampler. Kwargs hyperparameters are explained in its utils file.

    """

    @property
    def nf_model(self):
        return self.global_sampler.model

    def __init__(
        self,
        n_dim: int,
        rng_key_set: Tuple,
        data: jnp.ndarray,
        local_sampler: ProposalBase,
        nf_model: NFModel,
        adaptive_step_size: bool = True,
        **kwargs,
    ):
        rng_key_init, rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf = rng_key_set

        # Copying input into the model

        self.rng_keys_nf = rng_keys_nf
        self.rng_keys_mcmc = rng_keys_mcmc
        self.n_dim = n_dim
        self.adaptive_step_size = adaptive_step_size
        print("Adaptive step size is set to: ", self.adaptive_step_size)

        # Set and override any given hyperparameters
        self.hyperparameters = flowmc_default_hyperparameters
        hyperparameter_names = list(flowmc_default_hyperparameters.keys())
        
        for key, value in kwargs.items():
            if key in hyperparameter_names:
                self.hyperparameters[key] = value
        for key, value in self.hyperparameters.items():
            setattr(self, key, value)
            
        self.variables = {"mean": None, "var": None}

        # Initialized local and global samplers
        self.local_sampler = local_sampler
        if self.precompile:
            self.local_sampler.precompilation(
                n_chains=self.n_chains, n_dims=n_dim, n_step=self.n_local_steps, data=data
            )

        self.global_sampler = NFProposal(self.local_sampler.logpdf, jit=self.local_sampler.jit, model=nf_model, n_sample_max=self.n_sample_max)

        self.likelihood_vec = self.local_sampler.logpdf_vmap

        tx = optax.chain(optax.clip(1.0),optax.adam(self.learning_rate, self.momentum))
        self.optim_state = tx.init(eqx.filter(self.nf_model, eqx.is_array))
        self.nf_training_loop, train_epoch, train_step = make_training_loop(tx)

        # Initialized result dictionaries
        pretraining = initialize_summary_dict(self)
        training = initialize_summary_dict(self, use_loss_vals=True)
        production = initialize_summary_dict(self)

        self.summary = {}
        self.summary["pretraining"] = pretraining
        self.summary["training"] = training
        self.summary["production"] = production

    def sample(self, initial_position: Array, data: dict):
        """
        Sample from the posterior using the local sampler.

        Args:
            initial_position (Device Array): Initial position.

        Returns:
            chains (Device Array): Samples from the posterior.
            nf_samples (Device Array): (n_nf_samples, n_dim)
            local_accs (Device Array): (n_chains, n_local_steps * n_loop)
            global_accs (Device Array): (n_chains, n_global_steps * n_loop)
            loss_vals (Device Array): (n_epoch * n_loop,)
        """

        # Note that auto-tune function needs to have the same number of steps
        # as the actual sampling loop to avoid recompilation.

        self.local_sampler_tuning(initial_position, data)
        last_step = initial_position
        
        if self.n_loop_pretraining > 0:
            last_step = self.pretraining_run(last_step, data)
        
        if self.use_global == True and self.n_loop_training > 0:
            last_step = self.global_sampler_tuning(last_step, data)

        if self.n_loop_production > 0:
            last_step = self.production_run(last_step, data)

    def sampling_loop(
        self, initial_position: jnp.array, data: jnp.array, training=False, pretraining=False
    ) -> jnp.array:
        """
        One sampling loop that iterate through the local sampler and potentially the global sampler.
        If training is set to True, the loop will also train the normalizing flow model.

        Args:
            initial_position (jnp.array): Initial position. Shape (n_chains, n_dim)
            training (bool, optional): Whether to train the normalizing flow model. Defaults to False.

        Returns:
            chains (jnp.array): Samples from the posterior. Shape (n_chains, n_local_steps + n_global_steps, n_dim)
        """

        production = False
        if training == True:
            summary_mode = "training"
        elif pretraining == True:
            summary_mode = "pretraining"
        else:
            summary_mode = "production"
            production = True

        # First run the local sampler
        self.rng_keys_mcmc, positions, log_prob, local_acceptance = self.local_sampler.sample(
            self.rng_keys_mcmc,
            self.n_local_steps,
            initial_position,
            data,
            verbose=self.verbose,
        )

        # Save local sampler states
        self.summary[summary_mode]["chains"] = jnp.append(
            self.summary[summary_mode]["chains"], positions[:, ::self.output_thinning], axis=1
        )
        self.summary[summary_mode]["log_prob"] = jnp.append(
            self.summary[summary_mode]["log_prob"], log_prob[:, ::self.output_thinning], axis=1
        )

        self.summary[summary_mode]["local_accs"] = jnp.append(
            self.summary[summary_mode]["local_accs"], local_acceptance[:, 1::self.output_thinning], axis=1
        )
        # Run global sampler (if specified)
        if self.use_global == True:
            if training == True:
                positions = self.summary["training"]["chains"][
                    :, :: self.train_thinning
                ]
                log_prob_output = self.summary["training"]["log_prob"][
                    :, :: self.train_thinning
                ]

                if self.keep_quantile > 0:
                    max_log_prob = jnp.max(log_prob_output, axis=1)
                    cut = jnp.quantile(max_log_prob, self.keep_quantile)
                    cut_chains = positions[max_log_prob > cut]
                else:
                    cut_chains = positions
                
                chain_size = cut_chains.shape[0] * cut_chains.shape[1]
                if chain_size > self.max_samples:
                    flat_chain = cut_chains[
                        :, -int(self.max_samples / self.n_chains) :
                    ].reshape(-1, self.n_dim)
                else:
                    flat_chain = cut_chains.reshape(-1, self.n_dim)

                if flat_chain.shape[0] < self.max_samples:
                    # This is to pad the training data to avoid recompilation.
                    flat_chain = jnp.repeat(
                        flat_chain,
                        (self.max_samples // flat_chain.shape[0]) + 1,
                        axis=0,
                    )
                    flat_chain = flat_chain[: self.max_samples]

                self.variables["mean"] = jnp.mean(flat_chain, axis=0)
                self.variables["cov"] = jnp.cov(flat_chain.T)
                self.global_sampler.model = eqx.tree_at(
                    lambda m: m._data_mean, self.nf_model, self.variables["mean"]
                )
                self.global_sampler.model = eqx.tree_at(
                    lambda m: m._data_cov, self.nf_model, self.variables["cov"]
                )

                self.rng_keys_nf, self.global_sampler.model, self.optim_state, loss_values = self.nf_training_loop(
                    self.rng_keys_nf,
                    self.nf_model,
                    flat_chain,
                    self.optim_state,
                    self.n_epochs,
                    self.batch_size,
                    self.verbose,
                )
                self.summary["training"]["loss_vals"] = jnp.append(
                    self.summary["training"]["loss_vals"],
                    loss_values.reshape(1, -1),
                    axis=0,
                )
            # end if training=True
            
            # Sample globally
            (
                self.rng_keys_nf,
                nf_chain,
                log_prob,
                log_prob_nf,
                global_acceptance,
            ) = self.global_sampler.sample(
                self.rng_keys_nf,
                self.n_global_steps,
                positions[:, -1],
                data,
                verbose = self.verbose
            )

            # TODO this is wrong? Save the global samples only during production, or during training if specified
            if True: #production or (training and self.save_global_samples_training):
                self.summary[summary_mode]["chains"] = jnp.append(
                    self.summary[summary_mode]["chains"], nf_chain[:, ::self.output_thinning], axis=1
                )
                self.summary[summary_mode]["log_prob"] = jnp.append(
                    self.summary[summary_mode]["log_prob"], log_prob[:, ::self.output_thinning], axis=1
                )

                self.summary[summary_mode]["global_accs"] = jnp.append(
                    self.summary[summary_mode]["global_accs"],
                    global_acceptance[:, 1::self.output_thinning],
                    axis=1,
                )
                
                # TODO verify correctness, Compute Gelman-Rubin R for post-run diagnosis
                chains = self.summary[summary_mode]["chains"]
                R = gelman_rubin(chains)
                R = jnp.reshape(R, (-1, 1))
                self.summary[summary_mode]["gelman_rubin"] = jnp.append(
                    self.summary[summary_mode]["gelman_rubin"], R, axis=1
                )

        # Finally, return all final chain's final positions
        last_step = self.summary[summary_mode]["chains"][:, -1]
        
        # Adjust step size during training, if specified    
        if training and self.adaptive_step_size:
            last_local_accs = jnp.mean(local_acceptance[:, 1::self.output_thinning])
            gamma = compute_gamma(last_local_accs)
            # self.local_sampler.params["step_size"] *= gamma
            # self.local_sampler.params["gamma_T"] *= gamma
            self.local_sampler.gamma_T *= gamma
            
            self.summary[summary_mode]["gamma"] = jnp.append(
                self.summary[summary_mode]["gamma"], jnp.array([self.local_sampler.gamma_T]), axis=0
            )
            
            # TODO for debugging
            self.summary[summary_mode]["dt_test"] = jnp.append(
                self.summary[summary_mode]["dt_test"], jnp.array([self.local_sampler.params["step_size"][0, 0]]), axis=0
            )
            
        return last_step

    def local_sampler_tuning(
        self, initial_position: jnp.array, data: jnp.array, max_iter: int = 100
    ):
        """
        Tuning the local sampler. This runs a number of iterations of the local sampler,
        and then uses the acceptance rate to adjust the local sampler parameters.
        Since this is mostly for a fast adaptation, we do not carry the sample state forward.
        Instead, we only adapt the sampler parameters using the initial position.

        Args:
            n_steps (int): Number of steps to run the local sampler.
            initial_position (Device Array): Initial position for the local sampler.
            max_iter (int): Number of iterations to run the local sampler.
        """
        if self.local_autotune is not None:
            print("Autotune found, start tuning sampler_params")
            kernel_vmap = self.local_sampler.kernel_vmap
            self.local_sampler.params = self.local_autotune(
                kernel_vmap,
                self.rng_keys_mcmc,
                initial_position,
                self.likelihood_vec(initial_position),
                data,
                self.local_sampler.params,
                max_iter,
            )
        else:
            print("No autotune found, use input sampler_params")

    def global_sampler_tuning(
        self, initial_position: jnp.ndarray, data: jnp.array
    ) -> jnp.array:
        """
        Tuning the global sampler. This runs both the local sampler and the global sampler,
        and train the normalizing flow on the run.
        To adapt the normalizing flow, we need to keep certain amount of the data generated during the sampling.
        The data is stored in the summary dictionary and can be accessed through the `get_sampler_state` method.
        This tuning run is meant to be followed by a production run as defined below.

        Args:
            initial_position (Device Array): Initial position for the sampler, shape (n_chains, n_dim)

        """
        print("Training normalizing flow")
        last_step = initial_position
        for _ in tqdm(
            range(self.n_loop_training),
            desc="Tuning global sampler",
        ):
            last_step = self.sampling_loop(last_step, data, training=True)
        return last_step

    def pretraining_run(
        self, initial_position: jnp.ndarray, data: jnp.array
    ) -> jnp.array:
        """
        Sampling procedure that takes place before starting training on NF.
        The data is stored in the summary dictionary and can be accessed through the `get_sampler_state` method.

        Args:
            initial_position (Device Array): Initial position for the sampler, shape (n_chains, n_dim)

        """
        print("Starting pretraining run")
        last_step = initial_position
        for _ in tqdm(
            range(self.n_loop_pretraining),
            desc="Pretraining run",
        ):
            last_step = self.sampling_loop(last_step, data, pretraining=True)
        return last_step

    def production_run(
        self, initial_position: jnp.ndarray, data: jnp.array
    ) -> jnp.array:
        """
        Sampling procedure that produce the final set of samples.
        The main difference between this and the global tuning step is
        we do not train the normalizing flow, omitting training allows to maintain detail balance.
        The data is stored in the summary dictionary and can be accessed through the `get_sampler_state` method.

        Args:
            initial_position (Device Array): Initial position for the sampler, shape (n_chains, n_dim)

        """
        print("Starting Production run")
        last_step = initial_position
        for _ in tqdm(
            range(self.n_loop_production),
            desc="Production run",
        ):
            last_step = self.sampling_loop(last_step, data)
        return last_step

    def get_sampler_state(self, which: str="production") -> dict:
        """
        Get the sampler state. There are two sets of sampler outputs one can get,
        the training set and the production set.
        The training set is produced during the global tuning step, and the production set
        is produced during the production run.
        Only the training set contains information about the loss function.
        Only the production set should be used to represent the final set of samples.

        Args:
            training (bool): Whether to get the training set sampler state. Defaults to False.

        """
        if which not in ["pretraining", "training", "production"]:
            raise ValueError("Get sampler state got incorrect key")
        else:
            return self.summary[which]

    def sample_flow(self, n_samples: int) -> jnp.ndarray:
        """
        Sample from the normalizing flow.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            Device Array: Samples generated using the normalizing flow.
        """

        samples = self.nf_model.sample(self.rng_keys_nf, n_samples)
        return samples

    def evaluate_flow(self, samples: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the log probability of the normalizing flow.

        Args:
            samples (Device Array): Samples to evaluate.

        Returns:
            Device Array: Log probability of the samples.
        """
        log_prob = self.nf_model.log_prob(samples)
        return log_prob

    def save_flow(self, path: str):
        """
        Save the normalizing flow to a file.

        Args:
            path (str): Path to save the normalizing flow.
        """
        self.nf_model.save_model(path)

    def load_flow(self, path: str):
        """
        Save the normalizing flow to a file.

        Args:
            path (str): Path to save the normalizing flow.
        """
        self.nf_model = self.nf_model.load_model(path)

    def reset(self):
        """
        Reset the sampler state.

        """
        training = {}
        training["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
        training["log_prob"] = jnp.empty((self.n_chains, 0))
        training["local_accs"] = jnp.empty((self.n_chains, 0))
        training["global_accs"] = jnp.empty((self.n_chains, 0))
        training["loss_vals"] = jnp.empty((0, self.n_epochs))

        production = {}
        production["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
        production["log_prob"] = jnp.empty((self.n_chains, 0))
        production["local_accs"] = jnp.empty((self.n_chains, 0))
        production["global_accs"] = jnp.empty((self.n_chains, 0))

        self.summary = {}
        self.summary["training"] = training
        self.summary["production"] = production

    def get_global_acceptance_distribution(self, n_bins: int = 10, training: bool = False) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
        """
        Get the global acceptance distribution as a histogram per epoch.

        Returns:
            axis (Device Array): Axis of the histogram.
            hist (Device Array): Histogram of the global acceptance distribution.
        """
        if training == True:
            n_loop = self.n_loop_training
            global_accs = self.summary["training"]["global_accs"]
        else:
            n_loop = self.n_loop_production
            global_accs = self.summary["production"]["global_accs"]

        hist = [np.histogram(global_accs[:, i*(self.n_global_steps//self.output_thinning-1): (i+1)*(self.n_global_steps//self.output_thinning-1)].mean(axis=1), bins=n_bins) for i in range(n_loop)]
        axis = np.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = np.array([hist[i][0] for i in range(n_loop)]).T
        return axis, hist

    def get_local_acceptance_distribution(self, n_bins: int = 10, training: bool = False) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
        """
        Get the local acceptance distribution as a histogram per epoch.

        Returns:
            axis (Device Array): Axis of the histogram.
            hist (Device Array): Histogram of the local acceptance distribution.
        """
        if training == True:
            n_loop = self.n_loop_training
            local_accs = self.summary["training"]["local_accs"]
        else:
            n_loop = self.n_loop_production
            local_accs = self.summary["production"]["local_accs"]

        hist = [np.histogram(local_accs[:, i*(self.n_local_steps//self.output_thinning-1): (i+1)*(self.n_local_steps//self.output_thinning-1)].mean(axis=1), bins=n_bins) for i in range(n_loop)]
        axis = np.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = np.array([hist[i][0] for i in range(n_loop)]).T
        return axis, hist

    def get_log_prob_distribution(self, n_bins: int = 10, training: bool = False) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
        """
        Get the log probability distribution as a histogram per epoch.

        Returns:
            axis (Device Array): Axis of the histogram.
            hist (Device Array): Histogram of the log probability distribution.
        """
        if training == True:
            n_loop = self.n_loop_training
            log_prob = self.summary["training"]["log_prob"]
        else:
            n_loop = self.n_loop_production
            log_prob = self.summary["production"]["log_prob"]

        hist = [np.histogram(log_prob[:, i*(self.n_local_steps//self.output_thinning-1): (i+1)*(self.n_local_steps//self.output_thinning-1)].mean(axis=1), bins=n_bins) for i in range(n_loop)]
        axis = np.array([hist[i][1][:-1] for i in range(n_loop)]).T
        hist = np.array([hist[i][0] for i in range(n_loop)]).T
        return axis, hist

    def save_summary(self, path: str):
        """
        Save the summary to a file.

        Args:
            path (str): Path to save the summary.
        """
        with open(path, "wb") as f:
            pickle.dump(self.summary, f)
            
    def _single_plot(self, data: dict, name: str, which: str = "training", **plotkwargs):
        # Get plot kwargs
        figsize = plotkwargs["figsize"] if "figsize" in plotkwargs else (12, 8)
        alpha = plotkwargs["alpha"] if "alpha" in plotkwargs else 1

        plotdata = data[name]        

        eps=1e-3
        plt.figure(figsize=figsize)
        if name == "gamma" or name == "dt_test":
            mean= plotdata
        else:
            mean, _ = get_mean_and_std_chains(plotdata)
        x = [i+1 for i in range(len(mean))]
        plt.plot(x, mean, linestyle="-", color="blue", alpha=alpha)
        plt.xlabel("Iteration")
        plt.ylabel(f"{name} ({which})")
        # Extras for some variables:
        if "acc" in name:
            plt.ylim(0-eps, 1+eps)
            
        if "local_acc" in name:
            # TODO change this to only be used if we have MALA
            plt.axhline(0.574, color="grey", linestyle="--")
            
        if "gelman_rubin" in name:
            plt.axhline(1.1) # usual threshold for Gelman-Rubin R
        plt.savefig(f"{self.outdir_name}{name}_{which}.png", bbox_inches='tight')

    def plot_summary(self, which: str = "training", **plotkwargs):
        
        # Choose the dataset
        data = self.get_sampler_state(which)
        keys = ["local_accs", "global_accs", "log_prob", "gelman_rubin", "gamma", "dt_test"]
        if which == "training":
            keys = keys + ["loss_vals"]
        
        for key in keys:
            self._single_plot(data, key, which, **plotkwargs)
            
    def plot_chains(self, outdir_name: str, which: str = "production", nb_samples: int = 10000) -> None:
        # TODO add that user can give labels
        
        supported = ["pretraining", "training", "production", "NF"]
        if which not in supported:
            print(f"{which} for which in plot_chains not recognized, aborting plot creation")
            return
        
        # Get desired samples
        if which == "NF":
            samples = self.Sampler.sample_flow(nb_samples)
        else:
            chains = self.get_sampler_state(which)["chains"]
            # TODO make less cumbersome
            samples = jnp.array([chains[:, :, i].flatten() for i in range(self.n_dim)])
        
        # Corner wants them as numpy array    
        samples = np.asarray(samples)
        
        # TODO debugging, remove later:
        print(jnp.shape(samples))
            
        # Make sure that we have the (n_samples, n_dim) format
        if jnp.shape(samples)[0] < jnp.shape(samples)[1]:
            samples = np.swapaxes(samples, 0, 1)
        
        # Get the desired labels
        n_samples, n_dim = np.shape(samples)
        if len(IMRPhenomD_labels) == n_dim:
            labels = IMRPhenomD_labels
        else:
            print("Could not infer labels, setting to None")
            labels = None
            
        # Plot chains
        name = outdir_name + f"samples_{which}.png"
        if self.verbose:
            print(f"Saving plot of chains to {name}")
        fig = corner.corner(samples, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
        fig.savefig(name, bbox_inches='tight')  
        
@jax.jit
def compute_gamma(acceptance_rate, target_rate=0.574, width=0.5):
    """Simple linear"""
    gamma = width * (acceptance_rate - target_rate) + 1
    """No change"""
    # gamma = 1
    return gamma