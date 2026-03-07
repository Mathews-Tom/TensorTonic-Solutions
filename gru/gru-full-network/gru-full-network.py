import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last)."""
        batch_size, seq_len, _ = X.shape

        # Initialize hidden state
        h_t = np.zeros((batch_size, self.hidden_dim), dtype=float)

        hidden_states = []

        for t in range(seq_len):
            x_t = X[:, t, :]

            # Reset and update gates use [h_prev, x_t]
            concat_rz = np.concatenate([h_t, x_t], axis=-1)
            r_t = sigmoid(concat_rz @ self.W_r.T + self.b_r)
            z_t = sigmoid(concat_rz @ self.W_z.T + self.b_z)

            # Candidate uses [r_t * h_prev, x_t]
            concat_h = np.concatenate([r_t * h_t, x_t], axis=-1)
            h_tilde = np.tanh(concat_h @ self.W_h.T + self.b_h)

            # Final hidden update
            h_t = z_t * h_t + (1.0 - z_t) * h_tilde

            hidden_states.append(h_t)

        # Stack hidden states: (N, T, H)
        H = np.stack(hidden_states, axis=1)

        # Output projection: (N, T, O)
        y = H @ self.W_y.T + self.b_y

        return y, h_t