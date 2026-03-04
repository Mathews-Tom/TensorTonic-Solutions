import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net decoder block (shape-only):
      1) Upsample x: (H,W) -> (2H,2W)
      2) Crop skip to match upsampled spatial size
      3) Concat on channels: C -> C_up + C_skip
      4) Two valid 3x3 convs: spatial dims -4 total, channels -> out_channels

    Returns:
      out: (B, H_up-4, W_up-4, out_channels)
    """
    B, H, W, Cx = x.shape

    # 1) Upsample spatial dims by 2
    H_up, W_up = 2 * H, 2 * W

    # 2+3) After cropping skip and concatenation, spatial dims are (H_up, W_up)
    # Channels after concat would be out_channels (from up-conv) + skip_channels.
    # But since this is shape-only and next convs set channels to out_channels,
    # we only need correct final output shape.

    # 4) Two unpadded 3x3 convs reduce H/W by 4 total
    H_out, W_out = H_up - 4, W_up - 4

    return np.zeros((B, H_out, W_out, out_channels))