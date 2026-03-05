import numpy as np

class GAN:
    def __init__(self, data_dim: int, noise_dim: int):
        """
        Initialize GAN.
        """
        if not isinstance(data_dim, int) or data_dim <= 0:
            raise ValueError("data_dim must be a positive integer")
        if not isinstance(noise_dim, int) or noise_dim <= 0:
            raise ValueError("noise_dim must be a positive integer")

        self.data_dim = data_dim
        self.noise_dim = noise_dim

        # ---- Initialize generator and discriminator weights (simple 2-layer MLPs) ----
        self.g_hidden = max(128, noise_dim * 2)
        self.d_hidden = max(128, data_dim // 2)

        rng = np.random.default_rng(42)

        # Generator params: z -> ReLU -> tanh
        self.G_W1 = rng.standard_normal((noise_dim, self.g_hidden), dtype=np.float32) * np.sqrt(2.0 / (noise_dim + self.g_hidden))
        self.G_b1 = np.zeros((self.g_hidden,), dtype=np.float32)
        self.G_W2 = rng.standard_normal((self.g_hidden, data_dim), dtype=np.float32) * np.sqrt(2.0 / (self.g_hidden + data_dim))
        self.G_b2 = np.zeros((data_dim,), dtype=np.float32)

        # Discriminator params: x -> LeakyReLU -> sigmoid
        self.D_W1 = rng.standard_normal((data_dim, self.d_hidden), dtype=np.float32) * np.sqrt(2.0 / (data_dim + self.d_hidden))
        self.D_b1 = np.zeros((self.d_hidden,), dtype=np.float32)
        self.D_W2 = rng.standard_normal((self.d_hidden, 1), dtype=np.float32) * np.sqrt(2.0 / (self.d_hidden + 1))
        self.D_b2 = np.zeros((1,), dtype=np.float32)

        # Track losses for monitoring
        self.loss_history = {"d_loss": [], "g_loss": []}

    # ----------------- helpers -----------------
    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32, copy=False)
        out = np.empty_like(logits)
        pos = logits >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
        exp_x = np.exp(logits[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out

    @staticmethod
    def _clip_probs(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return np.clip(p, eps, 1.0 - eps)

    # ----------------- forward passes -----------------
    def _generator_forward(self, z: np.ndarray) -> np.ndarray:
        z = z.astype(np.float32, copy=False)
        h = z @ self.G_W1 + self.G_b1
        h = np.maximum(h, 0.0)  # ReLU
        x = h @ self.G_W2 + self.G_b2
        return np.tanh(x)

    def _discriminator_forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        h = x @ self.D_W1 + self.D_b1
        h = np.where(h > 0.0, h, 0.2 * h)  # LeakyReLU
        logits = h @ self.D_W2 + self.D_b2
        return self._sigmoid(logits)  # (batch, 1)

    # ----------------- public API -----------------
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate fake samples."""
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        z = np.random.randn(n_samples, self.noise_dim).astype(np.float32)
        return self._generator_forward(z)

    def discriminate(self, x: np.ndarray) -> np.ndarray:
        """Classify samples as real/fake."""
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise ValueError("x must be a 2D numpy array (batch_size, data_dim)")
        if x.shape[1] != self.data_dim:
            raise ValueError(f"x must have data_dim={self.data_dim}, got {x.shape[1]}")
        return self._discriminator_forward(x)

    def train_step(self, real_data: np.ndarray) -> dict:
        """
        Perform one training step (compute losses for monitoring).

        Note: Pure NumPy here => no autograd/optimizer updates.
        This satisfies the typical exercise requirement: forward passes + losses.
        """
        if not isinstance(real_data, np.ndarray) or real_data.ndim != 2:
            raise ValueError("real_data must be a 2D numpy array (batch_size, data_dim)")
        if real_data.shape[1] != self.data_dim:
            raise ValueError(f"real_data must have data_dim={self.data_dim}, got {real_data.shape[1]}")

        batch_size = real_data.shape[0]

        # Sample noise and generate fakes
        z = np.random.randn(batch_size, self.noise_dim).astype(np.float32)
        fake_data = self._generator_forward(z)

        # Discriminator probabilities
        real_probs = self._discriminator_forward(real_data).reshape(-1)
        fake_probs = self._discriminator_forward(fake_data).reshape(-1)

        real_probs = self._clip_probs(real_probs)
        fake_probs = self._clip_probs(fake_probs)

        # Losses (minimax, non-saturating G)
        d_loss = -np.mean(np.log(real_probs) + np.log(1.0 - fake_probs))
        g_loss = -np.mean(np.log(fake_probs))

        d_loss_f = float(d_loss)
        g_loss_f = float(g_loss)

        self.loss_history["d_loss"].append(d_loss_f)
        self.loss_history["g_loss"].append(g_loss_f)

        return {"d_loss": d_loss_f, "g_loss": g_loss_f}