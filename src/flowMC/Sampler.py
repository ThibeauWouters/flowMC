import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Callable

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource


class Sampler:
    """
    Top level API that the users primarily interact with.

    Args:
        n_dim (int): Dimension of the parameter space.
        n_chains (int): Number of chains to sample.
        rng_key (PRNGKeyArray): Jax PRNGKey.
        logpdf (Callable[[Float[Array, "n_dim"], dict], Float): Log probability function.
        resources (dict[str, Resource]): Resources to be used by the sampler.
        strategies (list[Strategy]): List of strategies to be used by the sampler.
        verbose (bool): Whether to print out progress. Defaults to False.
        logging (bool): Whether to log the progress. Defaults to True.
        outdir (str): Directory to save the logs. Defaults to "./outdir/".
    """

    # Essential parameters
    n_dim: int
    n_chains: int
    rng_key: PRNGKeyArray
    logpdf: Callable[[Float[Array, "n_dim"], dict], Float]
    resources: dict[str, Resource]
    strategies: list[Strategy]

    # Logging hyperparameters
    verbose: bool = False
    logging: bool = True
    outdir: str = "./outdir/"

    def __init__(
        self,
        n_dim: int,
        n_chains: int,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        rng_key: PRNGKeyArray,
        resources: dict[str, Resource],
        strategies: list[Strategy], #TODO: Set this to defult if not provided
        **kwargs,
    ):
        # Copying input into the model

        self.n_dim = n_dim
        self.n_chains = n_chains
        self.logpdf = logpdf
        self.rng_key = rng_key
        self.resources = resources
        self.strategies = strategies

        # Set and override any given hyperparameters
        class_keys = list(self.__class__.__dict__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def sample(self, initial_position: Float[Array, "n_chains n_dim"], data: dict):
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

        initial_position = jnp.atleast_2d(initial_position) # type: ignore
        rng_key = self.rng_key
        last_step = initial_position
        for strategy in self.strategies:
            (
                rng_key,
                self.resources,
                last_step,
            ) = strategy(
                rng_key, self.resources, last_step, data
            )


    # TODO: Move flow related function to flow class
    # def sample_flow(
    #     self, rng_key: PRNGKeyArray, n_samples: int
    # ) -> Float[Array, "n_samples n_dim"]:
    #     """
    #     Sample from the normalizing flow.

    #     Args:
    #         n_samples (int): Number of samples to generate.

    #     Returns:
    #         Device Array: Samples generated using the normalizing flow.
    #     """

    #     samples = self.nf_model.sample(rng_key, n_samples)
    #     return samples

    # def evalulate_flow(
    #     self, samples: Float[Array, "n_samples n_dim"]
    # ) -> Float[Array, "n_samples"]:
    #     """
    #     Evaluate the log probability of the normalizing flow.

    #     Args:
    #         samples (Device Array): Samples to evaluate.

    #     Returns:
    #         Device Array: Log probability of the samples.
    #     """
    #     log_prob = self.nf_model.log_prob(samples)
    #     return log_prob

    # def save_flow(self, path: str):
    #     """
    #     Save the normalizing flow to a file.

    #     Args:
    #         path (str): Path to save the normalizing flow.
    #     """
    #     self.nf_model.save_model(path)

    # def load_flow(self, path: str):
    #     """
    #     Save the normalizing flow to a file.

    #     Args:
    #         path (str): Path to save the normalizing flow.
    #     """
    #     self.global_sampler.model = self.nf_model.load_model(path)

    # TODO: Implement quick access and summary functions that operates on buffer

    # def get_sampler_state(self, training: bool = False) -> dict:
    #     """
    #     Get the sampler state. There are two sets of sampler outputs one can get,
    #     the training set and the production set.
    #     The training set is produced during the global tuning step, and the production set
    #     is produced during the production run.
    #     Only the training set contains information about the loss function.
    #     Only the production set should be used to represent the final set of samples.

    #     Args:
    #         training (bool): Whether to get the training set sampler state. Defaults to False.

    #     """
    #     if training is True:
    #         return self.summary["GlobalTuning"]
    #     else:
    #         return self.summary["GlobalSampling"]

    # def get_global_acceptance_distribution(
    #     self, n_bins: int = 10, training: bool = False
    # ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
    #     """
    #     Get the global acceptance distribution as a histogram per epoch.

    #     Returns:
    #         axis (Device Array): Axis of the histogram.
    #         hist (Device Array): Histogram of the global acceptance distribution.
    #     """
    #     if training is True:
    #         n_loop = self.n_loop_training
    #         global_accs = self.summary["training"]["global_accs"]
    #     else:
    #         n_loop = self.n_loop_production
    #         global_accs = self.summary["production"]["global_accs"]

    #     hist = [
    #         jnp.histogram(
    #             global_accs[
    #                 :,
    #                 i
    #                 * (self.n_global_steps // self.output_thinning - 1) : (i + 1)
    #                 * (self.n_global_steps // self.output_thinning - 1),
    #             ].mean(axis=1),
    #             bins=n_bins,
    #         )
    #         for i in range(n_loop)
    #     ]
    #     axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
    #     hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
    #     return axis, hist

    # def get_local_acceptance_distribution(
    #     self, n_bins: int = 10, training: bool = False
    # ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
    #     """
    #     Get the local acceptance distribution as a histogram per epoch.

    #     Returns:
    #         axis (Device Array): Axis of the histogram.
    #         hist (Device Array): Histogram of the local acceptance distribution.
    #     """
    #     if training is True:
    #         n_loop = self.n_loop_training
    #         local_accs = self.summary["training"]["local_accs"]
    #     else:
    #         n_loop = self.n_loop_production
    #         local_accs = self.summary["production"]["local_accs"]

    #     hist = [
    #         jnp.histogram(
    #             local_accs[
    #                 :,
    #                 i
    #                 * (self.n_local_steps // self.output_thinning - 1) : (i + 1)
    #                 * (self.n_local_steps // self.output_thinning - 1),
    #             ].mean(axis=1),
    #             bins=n_bins,
    #         )
    #         for i in range(n_loop)
    #     ]
    #     axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
    #     hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
    #     return axis, hist

    # def get_log_prob_distribution(
    #     self, n_bins: int = 10, training: bool = False
    # ) -> tuple[Int[Array, "n_bin n_loop"], Float[Array, "n_bin n_loop"]]:
    #     """
    #     Get the log probability distribution as a histogram per epoch.

    #     Returns:
    #         axis (Device Array): Axis of the histogram.
    #         hist (Device Array): Histogram of the log probability distribution.
    #     """
    #     if training is True:
    #         n_loop = self.n_loop_training
    #         log_prob = self.summary["training"]["log_prob"]
    #     else:
    #         n_loop = self.n_loop_production
    #         log_prob = self.summary["production"]["log_prob"]

    #     hist = [
    #         jnp.histogram(
    #             log_prob[
    #                 :,
    #                 i
    #                 * (self.n_local_steps // self.output_thinning - 1) : (i + 1)
    #                 * (self.n_local_steps // self.output_thinning - 1),
    #             ].mean(axis=1),
    #             bins=n_bins,
    #         )
    #         for i in range(n_loop)
    #     ]
    #     axis = jnp.array([hist[i][1][:-1] for i in range(n_loop)]).T
    #     hist = jnp.array([hist[i][0] for i in range(n_loop)]).T
    #     return axis, hist

    # def save_summary(self, path: str):
    #     """
    #     Save the summary to a file.

    #     Args:
    #         path (str): Path to save the summary.
    #     """
    #     with open(path, "wb") as f:
    #         pickle.dump(self.summary, f)

    # def print_summary(self) -> None:
    #     """
    #     Print summary to the screen about log probabilities and local/global acceptance rates.
    #     """
    #     train_summary = self.get_sampler_state(training=True)
    #     production_summary = self.get_sampler_state(training=False)

    #     training_log_prob = train_summary["log_prob"]
    #     training_local_acceptance = train_summary["local_accs"]
    #     training_global_acceptance = train_summary["global_accs"]
    #     training_loss = train_summary["loss_vals"]

    #     production_log_prob = production_summary["log_prob"]
    #     production_local_acceptance = production_summary["local_accs"]
    #     production_global_acceptance = production_summary["global_accs"]

    #     print("Training summary")
    #     print("=" * 10)
    #     print(
    #         f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}"
    #     )
    #     print(
    #         f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
    #     )
    #     print(
    #         f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
    #     )
    #     print(
    #         f"Max loss: {training_loss.max():.3f}, Min loss: {training_loss.min():.3f}"
    #     )

    #     print("Production summary")
    #     print("=" * 10)
    #     print(
    #         f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}"
    #     )
    #     print(
    #         f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
    #     )
    #     print(
    #         f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}"
    #     )