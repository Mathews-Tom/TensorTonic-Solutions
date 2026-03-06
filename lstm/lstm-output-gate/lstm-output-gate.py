import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    """Compute output gate o_t and hidden state h_t."""

    # Concatenate previous hidden state and input
    concat = np.concatenate([h_prev, x_t], axis=-1)   # (N, H + D)

    # Output gate
    o_t = sigmoid(concat @ W_o.T + b_o)               # (N, H)

    # Hidden state
    h_t = o_t * np.tanh(C_t)                          # (N, H)

    return o_t, h_t