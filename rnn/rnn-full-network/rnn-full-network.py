import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).

        X:   (N, T, input_dim)
        h_0: (N, hidden_dim) or None -> zeros
        y_seq: (N, T, output_dim)
        h_final: (N, hidden_dim)
        """
        N, T, _ = X.shape

        # init hidden
        h_t = np.zeros((N, self.hidden_dim), dtype=float) if h_0 is None else h_0

        # unroll RNN
        h_states = []
        for t in range(T):
            x_t = X[:, t, :]
            h_t = np.tanh((x_t @ self.W_xh.T) + (h_t @ self.W_hh.T) + self.b_h)
            h_states.append(h_t)

        h_all = np.stack(h_states, axis=1)      # (N, T, H)
        h_final = h_states[-1]                  # (N, H)

        # output projection for all time steps (efficient reshape)
        h_flat = h_all.reshape(N * T, self.hidden_dim)          # (N*T, H)
        y_flat = (h_flat @ self.W_hy.T) + self.b_y              # (N*T, O)
        y_seq = y_flat.reshape(N, T, -1)                        # (N, T, O)

        return y_seq, h_final