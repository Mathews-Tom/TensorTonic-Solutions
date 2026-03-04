import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.

        Supports:
          - x shape (B, C)
          - x shape (B, C, H, W)  (ResNet-style: normalize per-channel over B,H,W)
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 2:
            # Normalize per feature over batch
            axes = (0,)
            reshape = (1, -1)
        elif x.ndim == 4:
            # Normalize per channel over batch and spatial dims
            axes = (0, 2, 3)
            reshape = (1, -1, 1, 1)
        else:
            raise ValueError(f"BatchNorm expects 2D or 4D input, got shape {x.shape}")

        if training:
            batch_mean = np.mean(x, axis=axes)
            batch_var = np.var(x, axis=axes)

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * batch_var

            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        # Normalize
        x_hat = (x - mean.reshape(reshape)) / np.sqrt(var.reshape(reshape) + self.eps)

        # Scale and shift
        out = x_hat * self.gamma.reshape(reshape) + self.beta.reshape(reshape)
        return out

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    Works for:
      - x shape (B, C)
      - x shape (B, C, H, W) (applies linear map on channel dim per location)
    """
    x = np.asarray(x)

    if x.ndim == 2:
        h = x @ W1
        h = relu(bn1.forward(h, training=True))
        h = h @ W2
        h = relu(bn2.forward(h, training=True))
        return h

    if x.ndim == 4:
        # (B, C, H, W) -> (B, H, W, C)
        xp = x.transpose(0, 2, 3, 1)
        h = xp @ W1
        # BN expects (B, C, H, W)
        h = bn1.forward(h.transpose(0, 3, 1, 2), training=True)
        h = relu(h).transpose(0, 2, 3, 1)  # back to (B, H, W, C)

        h = h @ W2
        h = bn2.forward(h.transpose(0, 3, 1, 2), training=True)
        h = relu(h)

        return h  # (B, C, H, W)

    raise ValueError(f"post_activation_block expects 2D or 4D input, got shape {x.shape}")

def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    Uses x @ W for "convolution" (simplified as linear transform).
    Works for:
      - x shape (B, C)
      - x shape (B, C, H, W)
    """
    x = np.asarray(x)

    if x.ndim == 2:
        h = relu(bn1.forward(x, training=True))
        h = h @ W1
        h = relu(bn2.forward(h, training=True))
        h = h @ W2
        return h

    if x.ndim == 4:
        # BN + ReLU on (B, C, H, W)
        h = relu(bn1.forward(x, training=True))          # (B, C, H, W)
        hp = h.transpose(0, 2, 3, 1)                      # (B, H, W, C)
        hp = hp @ W1                                      # (B, H, W, C2)

        h2 = hp.transpose(0, 3, 1, 2)                     # (B, C2, H, W)
        h2 = relu(bn2.forward(h2, training=True))         # (B, C2, H, W)
        h2p = h2.transpose(0, 2, 3, 1)                    # (B, H, W, C2)
        h2p = h2p @ W2                                    # (B, H, W, C_out)

        return h2p.transpose(0, 3, 1, 2)                  # (B, C_out, H, W)

    raise ValueError(f"pre_activation_block expects 2D or 4D input, got shape {x.shape}")