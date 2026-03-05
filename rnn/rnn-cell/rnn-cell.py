import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.

    Computes:
        h_t = tanh(h_prev @ W_hh.T + x_t @ W_xh.T + b_h)

    Shapes:
        x_t:   (B, I)
        h_prev:(B, H)
        W_xh:  (H, I)
        W_hh:  (H, H)
        b_h:   (H,)
        h_t:   (B, H)
    """
    # Linear transforms (weights are (out, in), so use transpose)
    preact = (h_prev @ W_hh.T) + (x_t @ W_xh.T) + b_h  # b_h broadcasts over batch
    return np.tanh(preact)