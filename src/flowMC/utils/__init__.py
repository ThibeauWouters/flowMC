import jax.numpy as jnp
from jaxtyping import Array, Int, Float
import jax
import jax.numpy as jnp

def gelman_rubin(chains: Float[Array, "n_chains n_steps n_dim"], discard_fraction: float = 0.1) -> Array:
    """
    Static version of gelman rubin
    """
    _, _, n_dim = jnp.shape(chains)
    
    R_list = []
    
    for i in range(n_dim):
        # Get shape of chains for this parameter
        samples = chains[:, :, i]
        n_chains, length_chain = jnp.shape(samples)
        # Discard burn-in
        start_index = int(jnp.round(discard_fraction * length_chain))
        cut_samples = samples[:, start_index:]
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
            print(f"Error when computer Gelman-Rubin R statistic.")
            R = jnp.nan
        
        R = float(R)
        R_list.append(R)
    
    R_list = jnp.array(R_list)
    return R_list

def _compute_autocorrelation_time_single_chain(chain: Array, M: int=5, K: int=2, maxACL: int = 0):
    # TODO do we have to subtract the mean?
    # TODO what is a good max ACL?
    
    nPoints = len(chain)
    imax = nPoints / K
    
    lag = 1
    s = 1.0 / M
    cumACF = 1.0
    
    while cumACF >= s:
        x = chain[:-lag]
        y = chain[lag:]
        ACF = jnp.sum((x - jnp.mean(x)) * (y - jnp.mean(y))) / (jnp.std(x) * jnp.std(y))
        cumACF += 2.0 * ACF
        lag += 1
        s = lag/M
        if lag > imax:
            return jnp.inf
    ACL = cumACF 
    
    # TODO think about this one
    if ACL > maxACL:
        maxACL = ACL
    if jnp.isnan(ACL):
        return jnp.inf
    
    return maxACL

def compute_autocorrelation_time(chains: Array, M: int=5, K: int=2, maxACL: int = 0):
    
    batched_compute_autocorrelation = lambda x: _compute_autocorrelation_time_single_chain(x, M, K, maxACL)
    # result = jax.vmap(batched_compute_autocorrelation)(chains)
    # TODO convert to vmap result
    result = []
    for chain in chains:
        result.append(batched_compute_autocorrelation(chain))
    return result    


# TODO where to be used?
def discard_burn_in(chains: Array, log_prob: Array, n_dim: int):
    """TODO has still to be tested.
    To be used only in postproduction"""

    def _get_burn_in_start_single_chain(log_prob: Array, n_dim: int):
        """
        Watch out: log prob now represents values for a single chain
        """
        
        max_logL = jnp.max(log_prob)
        delta_logL = max_logL - n_dim/2.0
        idx = jnp.argmax(log_prob > delta_logL)
        
        return idx
    
    idx_array = jax.vmap(_get_burn_in_start_single_chain, in_axes=(0, None), out_axes=0)(log_prob, n_dim)
    print("idx_array (burn_in)")
    print(idx_array)
    new_chains = chains[:, idx_array:]
    
    return new_chains
    

@staticmethod
def get_mean_and_std_chains(chains: Array):
    """
    Chains has size (n_chains, length_data), average out the n_chains dimension.
    """
    
    return jnp.mean(chains, axis=0), jnp.std(chains, axis=0)

def initialize_summary_dict(sampler, use_loss_vals = False):
        
    my_dict = dict()

    my_dict["chains"] = jnp.empty((sampler.n_chains, 0, sampler.n_dim))
    my_dict["log_prob"] = jnp.empty((sampler.n_chains, 0))
    my_dict["local_accs"] = jnp.empty((sampler.n_chains, 0))
    my_dict["global_accs"] = jnp.empty((sampler.n_chains, 0))
    my_dict["gelman_rubin"] = jnp.empty((sampler.n_dim, 0))
    my_dict["gamma"] = jnp.empty(0) # single scalar, the rescaling factor for step size/mass matrix
    my_dict["dt_test"] = jnp.empty(0) # TODO remove, for debugging
    
    if use_loss_vals:
        my_dict["loss_vals"] = jnp.empty((0, sampler.n_epochs))
        
    return my_dict
    
    
default_corner_kwargs = dict(bins=40, 
                    smooth=1., 
                    show_titles=False,
                    label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16), 
                    color="blue",
                    # quantiles=[],
                    # levels=[0.9],
                    plot_density=True, 
                    plot_datapoints=False, 
                    fill_contours=True,
                    max_n_ticks=4, 
                    min_n_ticks=3,
                    save=False)

IMRPhenomD_labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}/{\rm Mpc}$', r"$t_c$"
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']