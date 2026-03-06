import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step (tanh RNN).

    Forward: h_t = tanh(z_t), where z_t = h_prev @ W_hh.T + x_t @ W_xh.T + b_h

    Given upstream gradient dh_next = dL/dh_t, compute:
      dL/dz_t = (1 - h_t^2) * dh_next
      dW_hh   = (dL/dz_t)^T @ h_prev
      dh_prev = (dL/dz_t) @ W_hh

    Args:
        dh_next: (B, H)
        h_t:     (B, H)
        h_prev:  (B, H)
        x_t:     (B, I)   # unused here since we only return dW_hh and dh_prev
        W_hh:    (H, H)

    Returns:
        dh_prev: (B, H)
        dW_hh:   (H, H)
    """
    # gradient through tanh
    dz = (1.0 - h_t ** 2) * dh_next  # (B, H)

    # gradients for recurrent weights and previous hidden state
    dW_hh = dz.T @ h_prev            # (H, H)
    dh_prev = dz @ W_hh              # (B, H)

    return dh_prev, dW_hh