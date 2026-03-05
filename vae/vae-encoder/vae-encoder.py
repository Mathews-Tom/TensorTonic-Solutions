import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters (mu, logvar).

    Args:
        x: input array of shape (B, D) (can be list -> converted)
        latent_dim: size of latent vector

    Returns:
        mu:     (B, latent_dim)
        logvar: (B, latent_dim)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D (B,D), got shape {x.shape}")

    B, D = x.shape

    # Simple 2-layer MLP: x -> h -> (mu, logvar)
    hidden_dim = 128

    W1 = np.random.randn(D, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    h = x @ W1 + b1
    h = np.maximum(0, h)  # ReLU

    W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
    b_mu = np.zeros(latent_dim)
    mu = h @ W_mu + b_mu

    W_lv = np.random.randn(hidden_dim, latent_dim) * 0.01
    b_lv = np.zeros(latent_dim)
    logvar = h @ W_lv + b_lv

    return mu, logvar