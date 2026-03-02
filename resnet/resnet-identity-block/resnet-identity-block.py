import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass (matches spec):
        y = ReLU((ReLU(x @ W1)) @ W2) + x   for 2D
        and per-spatial-location for 4D.
        """
        x = np.asarray(x)

        # (C,) -> treat as (1, C)
        squeeze_back = False
        if x.ndim == 1:
            x = x[None, :]
            squeeze_back = True

        if x.ndim == 2:
            # (B, C)
            h = relu(x @ self.W1)
            h = relu(h @ self.W2)
            y = h + x
            return y[0] if squeeze_back else y

        if x.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            x_perm = x.transpose(0, 2, 3, 1)
            h = relu(x_perm @ self.W1)
            h = relu(h @ self.W2)
            y = h + x_perm
            return y.transpose(0, 3, 1, 2)

        raise ValueError(f"Expected x to be 1D, 2D, or 4D; got shape {x.shape}")