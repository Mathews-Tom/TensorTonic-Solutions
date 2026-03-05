import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    mu = np.asarray(mu)
    log_var = np.asarray(log_var)

    std = np.exp(0.5 * log_var)
    eps = np.random.randn(*mu.shape)

    z = mu + std * eps
    return z