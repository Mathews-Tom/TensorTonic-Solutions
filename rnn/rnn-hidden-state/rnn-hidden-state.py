import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Initialize the hidden state for an RNN.

    Returns:
        h0 with shape (batch_size, hidden_dim) filled with zeros.
    """
    return np.zeros((batch_size, hidden_dim), dtype=float)