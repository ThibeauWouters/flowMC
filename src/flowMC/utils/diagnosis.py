from typing import Callable
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm
from flowMC.sampler.Proposal_Base import ProposalBase
from functools import partialmethod
from jaxtyping import PyTree, Array, Float, Int, PRNGKeyArray


class Diagnosis:
    
    """Class to run diagnosis on your MCMC chains"""
    
    # TODO add type hinting here
    def __init__(self, chains: dict) -> None:
        
        self.chains = chains
        self.param_names = list(self.chains.keys())
        
    def gelman_rubin(self, discard_fraction: float = 0.5):
        """
        See https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/bayespputils.py#L1988C14-L1988C14
        
        Source: 
        Brooks, Stephen P., and Andrew Gelman. 
        "General methods for monitoring convergence of iterative simulations." 
        Journal of computational and graphical statistics 7.4 (1998): 434-455.
        """
        
        result = dict()
        pnames = list(self.chains.keys())
        
        for pname in pnames:
            # Get shape of chains for this parameter
            samples = self.chains[pname]
            n_chains, length_chain = jnp.shape(samples)
            
            # Discard burn-in
            start_index = int(jnp.round(discard_fraction * length_chain))
            samples = samples[:, start_index:]
            
            # Do Gelman-Rubin statistic computation
            chain_means = jnp.mean(samples, axis=1)
            # print("Shape chain means")
            # print(jnp.shape(chain_means))
            chain_vars = jnp.var(samples, axis=1)
            # print("Shape chain vars")
            # print(jnp.shape(chain_vars))
            BoverN = jnp.var(chain_means)
            W = jnp.mean(chain_vars)
            
            sigmaHat2 = W + BoverN
            m = n_chains
            VHat = sigmaHat2 + BoverN/m
        
            try:
                R = VHat/W
            except:
                print(f"Error when computer Gelman-Rubin R statistic for {pname}.  This may be a fixed parameter")
                R = jnp.nan
            
            result[pname] = float(R)
        
        return result