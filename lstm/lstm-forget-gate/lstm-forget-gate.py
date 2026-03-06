import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h_prev, x_t] + b_f)"""

    # concatenate hidden state and input
    concat = np.concatenate([h_prev, x_t], axis=-1)   # (N, H + D)

    # linear transform + bias
    z = concat @ W_f.T + b_f                          # (N, H)

    # sigmoid activation
    f_t = sigmoid(z)

    return f_t