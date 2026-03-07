import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """Compute candidate: h_tilde = tanh(W_h @ [r_t * h_prev, x_t] + b_h)"""

    # Apply reset gate to previous hidden state
    gated_h = r_t * h_prev                       # (N, H)

    # Concatenate gated hidden state with input
    concat = np.concatenate([gated_h, x_t], axis=-1)  # (N, H + D)

    # Linear transformation + tanh activation
    h_tilde = np.tanh(concat @ W_h.T + b_h)      # (N, H)

    return h_tilde