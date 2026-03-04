import numpy as np

def relu(x):
    return np.maximum(0, x)

class BottleneckBlock:
    """
    Bottleneck Block: 1x1 -> 3x3 -> 1x1
    Reduces computation by compressing channels.
    """
    
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels  # Compressed dimension
        self.out_ch = out_channels
        
        # 1x1 reduce
        self.W1 = np.random.randn(in_channels, bottleneck_channels) * 0.01
        # 3x3 (simplified as dense)
        self.W2 = np.random.randn(bottleneck_channels, bottleneck_channels) * 0.01
        # 1x1 expand
        self.W3 = np.random.randn(bottleneck_channels, out_channels) * 0.01
        
        # Shortcut (if dimensions differ)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01 if in_channels != out_channels else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Bottleneck forward: compress -> process -> expand + skip

        Supports:
          - x shape (C,) or (B, C)
          - x shape (B, C, H, W) (applies per spatial location)
        """
        x = np.asarray(x)

        squeeze_back = False
        if x.ndim == 1:
            x = x[None, :]
            squeeze_back = True

        if x.ndim == 2:
            # Main path
            h = relu(x @ self.W1)          # (B, bn_ch)
            h = relu(h @ self.W2)          # (B, bn_ch)
            h = h @ self.W3                # (B, out_ch)

            # Shortcut
            shortcut = x if self.Ws is None else (x @ self.Ws)

            y = h + shortcut
            return y[0] if squeeze_back else y

        if x.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            x_perm = x.transpose(0, 2, 3, 1)

            # Main path per spatial location
            h = relu(x_perm @ self.W1)     # (B, H, W, bn_ch)
            h = relu(h @ self.W2)          # (B, H, W, bn_ch)
            h = h @ self.W3                # (B, H, W, out_ch)

            # Shortcut projection if needed
            shortcut = x_perm if self.Ws is None else (x_perm @ self.Ws)

            y = h + shortcut

            # Back to (B, out_ch, H, W)
            return y.transpose(0, 3, 1, 2)

        raise ValueError(f"Expected x to be 1D, 2D, or 4D; got shape {x.shape}")