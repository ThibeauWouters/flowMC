from typing import Callable
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm
from flowMC.sampler.Proposal_Base import ProposalBase
from flowMC.sampler.Sampler import Sampler
from functools import partialmethod
from jaxtyping import PyTree, Array, Float, Int, PRNGKeyArray

import matplotlib.pyplot as plt
import os

plotparams = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(plotparams)

class Diagnosis:
    
    """Class to run diagnosis on your MCMC chains"""
    
    # TODO add type hinting here
    def __init__(self, sampler: Sampler, naming=None, outdir_name: str = "./outdir/") -> None:
        
        self.sampler = sampler
        if naming is None:
            naming = [f"x{i}" for i in range(sampler.n_dim)]
        self.naming = naming
        self.train_summary = sampler.get_sampler_state(training=True)
        self.production_summary = sampler.get_sampler_state(training=False)
        
        print("self.train_summary.keys()")
        print(self.train_summary.keys())
        print("self.production_summary.keys()")
        print(self.production_summary.keys())
        
        self.outdir_name = outdir_name
        dir_exists = os.path.exists(outdir_name)
        if not dir_exists:
            os.makedirs(outdir_name)
        
    def _get_named_dictionary_chains(self, chains):
        """
        Convert chains from summaries to dictionaries with correct names
        """
        
        result = dict()
        for i, name in enumerate(self.naming):
            result[name] = chains[:, :, i]
        
        return result
    
    @staticmethod
    def _get_mean_and_std_chains(chains: Array):
        """
        Chains has size (n_chains, length_data).
        """
        
        return jnp.mean(chains, axis=0), jnp.std(chains, axis=0)
    
    @staticmethod
    def _rolling_average(x, N=10):
        # TODO fix such that rightmost boundary is more informative
        new_x = jnp.convolve(x, jnp.ones(N)/N)
        return new_x[N-1:] 

    
    def plot_summary(self, which: str = "training", **plotkwargs):
        
        # TODO add a plot of the Gelman-Rubin statistic
        
        # Choose the dataset
        if which == "training":
            data = self.train_summary
        else:
            data = self.production_summary
        
        local_accs = data["local_accs"]
        global_accs = data["global_accs"]
        log_prob = data["log_prob"]
        
        # Get plot kwargs
        figsize = plotkwargs["figsize"] if "figsize" in plotkwargs else (12, 8)
        alpha = plotkwargs["alpha"] if "alpha" in plotkwargs else 0.2
        
        # Local acceptance
        eps=1e-3
        plt.figure(figsize=figsize)
        mean, std = self._get_mean_and_std_chains(local_accs)
        x = [i for i in range(len(mean))]
        plt.plot(x, mean, linestyle="-", color="blue", alpha=alpha)
        plt.plot(x, self._rolling_average(mean), linestyle="-", color="blue")
        plt.xlabel("Iteration")
        plt.xscale('log')
        plt.ylabel(f"Local acceptance (f{which})")
        plt.ylim(0-eps, 1+eps)
        plt.savefig(self.outdir_name + f"local_accs_{which}.png", bbox_inches='tight')
        
        # Global acceptance
        plt.figure(figsize=figsize)
        mean, std = self._get_mean_and_std_chains(global_accs)
        x = [i for i in range(len(mean))]
        plt.plot(x, mean, linestyle="-", color="blue", alpha=alpha)
        plt.plot(x, self._rolling_average(mean), linestyle="-", color="blue")
        plt.xlabel("Iteration")
        plt.xscale('log')
        plt.ylabel(f"Global acceptance (f{which})")
        plt.ylim(0-eps, 1+eps)
        plt.savefig(self.outdir_name + f"global_accs_{which}.png", bbox_inches='tight')
        
        # Log prob
        plt.figure(figsize=figsize)
        mean, std = self._get_mean_and_std_chains(log_prob)
        x = [i for i in range(len(mean))]
        plt.plot(x, mean, linestyle="-", color="blue")
        plt.fill_between(x, y1=mean - std, y2=mean + std, color="blue", alpha=alpha)
        plt.xlabel("Iteration")
        plt.xscale('log')
        plt.ylabel(f"Log prob (f{which})")
        plt.savefig(self.outdir_name + f"log_prob_{which}.png", bbox_inches='tight')
        
        # For training, also plot the loss
        if which == "training":
            loss_vals = data["loss_vals"]    
            print(jnp.shape(loss_vals))
            mean, std = self._get_mean_and_std_chains(loss_vals)
            # Loss vals
            plt.figure(figsize=figsize)
            x = [i for i in range(len(mean))]
            plt.plot(x, mean, linestyle="-", color="blue")
            plt.fill_between(x, y1=mean - std, y2=mean + std, color="blue", alpha=alpha)
            plt.xlabel("Iteration")
            plt.xscale('log')
            plt.ylabel(f"Loss values (f{which})")
            plt.savefig(self.outdir_name + f"loss_nf.png", bbox_inches='tight')
            
        
    # TODO cumbersome implementation for now, can be improved in the future
    def gelman_rubin(self, discard_fraction: float = 0.1, n_points=10):
        """
        See https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/bayespputils.py#L1988C14-L1988C14
        
        Source: 
        Brooks, Stephen P., and Andrew Gelman. 
        "General methods for monitoring convergence of iterative simulations." 
        Journal of computational and graphical statistics 7.4 (1998): 434-455.
        """
        
        result = dict()
        chains = self.production_summary["chains"]
        chains = self._get_named_dictionary_chains(chains)
        
        for pname in self.naming:
            R_list = []
            # Get shape of chains for this parameter
            samples = chains[pname]
            n_chains, length_chain = jnp.shape(samples)
            # Discard burn-in
            start_index = int(jnp.round(discard_fraction * length_chain))
            # Remaining length gets subdivided into parts to check how the R value evolves over time
            remaining_length = length_chain - start_index - 1
            idx_list = jnp.round(jnp.linspace(0, remaining_length, n_points, endpoint=False)).astype(int)
            for idx in idx_list[1:]:
                cut_samples = samples[:, start_index:start_index+idx]
                # Do Gelman-Rubin statistic computation
                chain_means = jnp.mean(cut_samples, axis=1)
                chain_vars = jnp.var(cut_samples, axis=1)
                BoverN = jnp.var(chain_means)
                W = jnp.mean(chain_vars)
                sigmaHat2 = W + BoverN
                m = n_chains
                VHat = sigmaHat2 + BoverN/m
                try:
                    R = VHat/W
                except:
                    print(f"Error when computer Gelman-Rubin R statistic for {pname}.")
                    R = jnp.nan
                R_list.append(float(R))
                
            result[pname] = R_list
        
        return result