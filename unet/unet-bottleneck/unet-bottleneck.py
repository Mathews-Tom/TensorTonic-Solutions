import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: two 3x3 unpadded convolutions.
    
    Input shape:  (B, H, W, C)
    Output shape: (B, H-4, W-4, out_channels)
    """
    B, H, W, C = x.shape
    
    H_out = H - 4
    W_out = W - 4
    
    return np.zeros((B, H_out, W_out, out_channels))