import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """Compute update gate: z_t = sigmoid(W_z @ [h_prev, x_t] + b_z)"""

    # Concatenate previous hidden state and input
    concat = np.concatenate([h_prev, x_t], axis=-1)   # (N, H + D)

    # Linear transform + sigmoid
    z_t = sigmoid(concat @ W_z.T + b_z)               # (N, H)

    return z_t