import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last, C_last)."""
        batch_size, seq_len, _ = X.shape

        # Initialize hidden and cell states
        h_t = np.zeros((batch_size, self.hidden_dim), dtype=float)
        C_t = np.zeros((batch_size, self.hidden_dim), dtype=float)

        hidden_states = []

        for t in range(seq_len):
            x_t = X[:, t, :]

            # Concatenate h_{t-1} and x_t
            concat = np.concatenate([h_t, x_t], axis=-1)

            # Gates
            f_t = sigmoid(concat @ self.W_f.T + self.b_f)
            i_t = sigmoid(concat @ self.W_i.T + self.b_i)
            c_tilde = np.tanh(concat @ self.W_c.T + self.b_c)
            o_t = sigmoid(concat @ self.W_o.T + self.b_o)

            # Update cell and hidden states
            C_t = f_t * C_t + i_t * c_tilde
            h_t = o_t * np.tanh(C_t)

            hidden_states.append(h_t)

        # Stack hidden states: (N, T, H)
        H = np.stack(hidden_states, axis=1)

        # Output projection: (N, T, O)
        y = H @ self.W_y.T + self.b_y

        return y, h_t, C_t