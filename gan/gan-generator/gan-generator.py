import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.

    Args:
        z: Noise vectors of shape (batch_size, latent_dim)
        output_dim: Dimension of generated samples

    Returns:
        Fake samples of shape (batch_size, output_dim)
    """
    if not isinstance(z, np.ndarray):
        raise TypeError("z must be a numpy.ndarray")
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (batch_size, latent_dim), got shape {z.shape}")
    if not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError("output_dim must be a positive integer")

    batch_size, latent_dim = z.shape

    # Deterministic "network weights" so repeated calls are stable without storing parameters.
    # (In a real GAN, these would be learned parameters.)
    seed = (latent_dim * 1_000_003 + output_dim) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    hidden_dim = max(128, latent_dim * 2)

    # Xavier/Glorot-ish init scales
    w1 = rng.standard_normal((latent_dim, hidden_dim), dtype=np.float32) * np.sqrt(2.0 / (latent_dim + hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = rng.standard_normal((hidden_dim, output_dim), dtype=np.float32) * np.sqrt(2.0 / (hidden_dim + output_dim))
    b2 = np.zeros((output_dim,), dtype=np.float32)

    z = z.astype(np.float32, copy=False)

    # MLP: z -> ReLU -> output
    h = z @ w1 + b1
    h = np.maximum(h, 0.0)  # ReLU
    out = h @ w2 + b2

    # Squash to a bounded range (common for normalized data)
    fake_data = np.tanh(out)

    return fake_data