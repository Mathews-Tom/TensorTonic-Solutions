import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    Used when input/output dimensions differ.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path weights
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        
        # Shortcut projection (1x1 conv equivalent)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut.

        Spec: y = F(x) + Ws * x
        We use ReLU after each main-path layer (like the identity block prompt).
        """
        x = np.asarray(x)

        # (C,) -> (1, C)
        squeeze_back = False
        if x.ndim == 1:
            x = x[None, :]
            squeeze_back = True

        if x.ndim == 2:
            # (B, Cin)
            main = relu(x @ self.W1)
            main = relu(main @ self.W2)
            shortcut = x @ self.Ws
            y = main + shortcut
            return y[0] if squeeze_back else y

        if x.ndim == 4:
            # (B, Cin, H, W)
            # Downsample if stride > 1 (simple strided sampling)
            if self.stride > 1:
                x_ds = x[:, :, ::self.stride, ::self.stride]
            else:
                x_ds = x

            # Move channels last: (B, H, W, Cin)
            x_perm = x_ds.transpose(0, 2, 3, 1)

            # Main path: (B, H, W, Cout)
            main = relu(x_perm @ self.W1)
            main = relu(main @ self.W2)

            # Shortcut projection: (B, H, W, Cout)
            shortcut = x_perm @ self.Ws

            # Add skip (projection) connection
            y = main + shortcut

            # Back to (B, Cout, H, W)
            y = y.transpose(0, 3, 1, 2)
            return y

        raise ValueError(f"Expected x to be 1D, 2D, or 4D; got shape {x.shape}")