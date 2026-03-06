import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """Compute reset gate: r_t = sigmoid(W_r @ [h_prev, x_t] + b_r)"""

    # Concatenate previous hidden state and current input
    concat = np.concatenate([h_prev, x_t], axis=-1)   # (N, H + D)

    # Linear transformation + sigmoid
    r_t = sigmoid(concat @ W_r.T + b_r)               # (N, H)

    return r_t