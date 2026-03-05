import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.

    Args:
        z: latent array of shape (B, latent_dim) (list ok)
        output_dim: target reconstruction dimension D

    Returns:
        x_hat: reconstructed array of shape (B, output_dim)
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"Expected z to be 2D (B,latent_dim), got shape {z.shape}")

    B, latent_dim = z.shape
    hidden_dim = 128

    # z -> h
    W1 = np.random.randn(latent_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    h = z @ W1 + b1
    h = np.maximum(0, h)  # ReLU

    # h -> x_hat
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros(output_dim)
    x_hat = h @ W2 + b2

    return x_hat