import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.

    Shapes:
      x  : (batch, seq_len, d_model)
      W1 : (d_model, d_ff)
      b1 : (d_ff,)
      W2 : (d_ff, d_model)
      b2 : (d_model,)

    Returns:
      (batch, seq_len, d_model)
    """
    # First linear layer + bias
    hidden = x @ W1 + b1

    # ReLU activation
    hidden = np.maximum(0, hidden)

    # Second linear layer + bias
    out = hidden @ W2 + b2

    return out