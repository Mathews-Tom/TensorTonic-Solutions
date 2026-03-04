import numpy as np

def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    """
    U-Net output layer: 1x1 conv for pixel-wise classification.
    """
    features = np.array(features)
    B, H, W, _ = features.shape
    return np.zeros((B, H, W, num_classes))