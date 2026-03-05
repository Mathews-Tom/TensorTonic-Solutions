import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 128

        # Encoder weights: x -> h -> (mu, log_var)
        self.W1_enc = np.random.randn(input_dim, self.hidden_dim) * 0.01
        self.b1_enc = np.zeros(self.hidden_dim)

        self.W_mu = np.random.randn(self.hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)

        self.W_lv = np.random.randn(self.hidden_dim, latent_dim) * 0.01
        self.b_lv = np.zeros(latent_dim)

        # Decoder weights: z -> h -> x_hat
        self.W1_dec = np.random.randn(latent_dim, self.hidden_dim) * 0.01
        self.b1_dec = np.zeros(self.hidden_dim)

        self.W_out = np.random.randn(self.hidden_dim, input_dim) * 0.01
        self.b_out = np.zeros(input_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def _encode(self, x: np.ndarray) -> tuple:
        x = np.asarray(x, dtype=float)
        h = self._relu(x @ self.W1_enc + self.b1_enc)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_lv + self.b_lv
        return mu, log_var

    def _reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def _decode(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        h = self._relu(z @ self.W1_dec + self.b1_dec)
        x_hat = h @ self.W_out + self.b_out
        return x_hat

    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through VAE.
        Returns: (x_recon, mu, log_var)
        """
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        x_recon = self._decode(z)
        return x_recon, mu, log_var

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples from prior z ~ N(0, I).
        Returns: samples of shape (n_samples, input_dim)
        """
        z = np.random.randn(n_samples, self.latent_dim)
        return self._decode(z)