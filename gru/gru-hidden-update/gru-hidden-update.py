import numpy as np

def hidden_update(h_prev: np.ndarray, h_tilde: np.ndarray,
                  z_t: np.ndarray) -> np.ndarray:
    """Compute final state: h_t = z_t * h_prev + (1 - z_t) * h_tilde"""

    h_t = z_t * h_prev + (1 - z_t) * h_tilde
    return h_t