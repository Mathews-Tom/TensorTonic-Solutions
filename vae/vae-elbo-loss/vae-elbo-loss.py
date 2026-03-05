import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss:
      recon = ||x - x_recon||^2  (sum of squared error)
      kl    = D_KL(q(z|x) || N(0,I))
      total = recon + kl
    Returns dict with floats: {"total": ..., "recon": ..., "kl": ...}
    """
    x = np.asarray(x, dtype=float)
    x_recon = np.asarray(x_recon, dtype=float)
    mu = np.asarray(mu, dtype=float)
    log_var = np.asarray(log_var, dtype=float)

    recon = float(np.sum((x - x_recon) ** 2))
    kl = float(-0.5 * np.sum(1.0 + log_var - mu**2 - np.exp(log_var)))
    total = recon + kl

    return {"total": float(total), "recon": recon, "kl": kl}