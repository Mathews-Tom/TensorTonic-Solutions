import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    U-Net encoder block (shape-only):
      conv3x3 valid  -> H,W reduce by 2
      conv3x3 valid  -> H,W reduce by 2 again
      maxpool2x2     -> H,W halve

    Returns:
      pool_out: (B, (H-4)//2, (W-4)//2, out_channels)
      skip_out: (B, H-4, W-4, out_channels)
    """
    B, H, W, C = x.shape

    # After two valid 3x3 convs
    H2, W2 = H - 4, W - 4
    skip_out = np.zeros((B, H2, W2, out_channels))

    # After 2x2 max pool with stride 2
    pool_out = np.zeros((B, H2 // 2, W2 // 2, out_channels))

    return pool_out, skip_out