import numpy as np

def discriminator(x: np.ndarray) -> np.ndarray:
    """
    Classify inputs as real or fake.

    Args:
        x: Input data of shape (batch_size, input_dim)

    Returns:
        Probabilities of shape (batch_size, 1) with values in [0, 1]
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy.ndarray")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (batch_size, input_dim), got shape {x.shape}")

    batch_size, input_dim = x.shape

    # Deterministic "network weights" based on input_dim (no stored parameters).
    # (In a real GAN, these would be learned.)
    seed = (input_dim * 2_000_033 + 1337) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    hidden_dim = max(128, input_dim // 2)

    w1 = rng.standard_normal((input_dim, hidden_dim), dtype=np.float32) * np.sqrt(2.0 / (input_dim + hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = rng.standard_normal((hidden_dim, 1), dtype=np.float32) * np.sqrt(2.0 / (hidden_dim + 1))
    b2 = np.zeros((1,), dtype=np.float32)

    x = x.astype(np.float32, copy=False)

    # MLP: x -> LeakyReLU -> sigmoid
    h = x @ w1 + b1
    h = np.where(h > 0.0, h, 0.2 * h)  # LeakyReLU with slope 0.2
    logits = h @ w2 + b2

    # Stable sigmoid
    logits = logits.astype(np.float32, copy=False)
    probs = np.empty_like(logits)
    pos = logits >= 0
    probs[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_x = np.exp(logits[~pos])
    probs[~pos] = exp_x / (1.0 + exp_x)

    return probs.reshape(batch_size, 1)