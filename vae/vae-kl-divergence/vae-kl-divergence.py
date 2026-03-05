import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    """
    mu = np.asarray(mu)
    log_var = np.asarray(log_var)

    kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    return float(kl)