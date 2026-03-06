import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.

    Computes for t=0..T-1:
        h_t = tanh(h_{t-1} @ W_hh.T + x_t @ W_xh.T + b_h)

    Args:
        X:   (batch, time, input_dim)
        h_0: (batch, hidden_dim)
        W_xh:(hidden_dim, input_dim)
        W_hh:(hidden_dim, hidden_dim)
        b_h: (hidden_dim,)

    Returns:
        h_all:   (batch, time, hidden_dim)  # all hidden states stacked along time axis
        h_final: (batch, hidden_dim)        # last hidden state
    """
    h_t = h_0
    states = []

    # iterate over time dimension
    for t in range(X.shape[1]):
        x_t = X[:, t, :]
        h_t = np.tanh((h_t @ W_hh.T) + (x_t @ W_xh.T) + b_h)
        states.append(h_t)

    h_all = np.stack(states, axis=1)  # (B, T, H)
    h_final = states[-1]              # last state
    return h_all, h_final