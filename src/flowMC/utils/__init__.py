import jax.numpy as jnp
from jaxtyping import Array, Int, Float

def gelman_rubin(chains: Float[Array, "n_chains n_steps n_dim"], discard_fraction: float = 0.1):
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
    
    print("R_list")
    print(R_list)
    avg_R = jnp.mean(jnp.array(R_list))
    return avg_R