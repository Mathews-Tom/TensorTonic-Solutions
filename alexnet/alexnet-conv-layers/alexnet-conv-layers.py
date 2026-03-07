import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    
    N, H, W, C = image.shape
    
    k = 11
    s = 4
    p = 2
    filters = 96
    
    H_out = (H + 2*p - k) // s + 1
    W_out = (W + 2*p - k) // s + 1
    
    return np.zeros((N, H_out, W_out, filters))