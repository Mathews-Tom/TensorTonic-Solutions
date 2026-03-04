import numpy as np

def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    """
    Implement a VGG-style convolutional block (simplified).

    Assumes input is NHWC: (batch, H, W, C_in)
    Preserves H and W, changes channels to out_channels.
    Applies num_convs linear "conv" layers, each followed by ReLU.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 4:
        raise ValueError(f"Expected x to have shape (B,H,W,C), got {x.shape}")
    if num_convs < 1:
        return x

    B, H, W, C_in = x.shape
    h = x

    in_ch = C_in
    for _ in range(num_convs):
        # "3x3 conv" simplified as per-pixel linear projection on channels
        W = np.random.randn(in_ch, out_channels) * 0.01
        h = h @ W                      # (B,H,W,out_channels)
        h = np.maximum(0, h)           # ReLU
        in_ch = out_channels           # next layer input channels

    return h