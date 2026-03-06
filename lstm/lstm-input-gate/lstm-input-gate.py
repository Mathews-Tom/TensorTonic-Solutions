import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate i_t and candidate memory C_tilde."""

    # Concatenate hidden state and input
    concat = np.concatenate([h_prev, x_t], axis=-1)   # (N, H + D)

    # Input gate
    i_t = sigmoid(concat @ W_i.T + b_i)               # (N, H)

    # Candidate memory
    c_tilde = np.tanh(concat @ W_c.T + b_c)           # (N, H)

    return i_t, c_tilde