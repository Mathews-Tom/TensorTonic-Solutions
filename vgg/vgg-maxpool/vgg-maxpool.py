import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).

    Input:  (B, H, W, C)
    Output: (B, H//2, W//2, C)
    """
    B, H, W, C = x.shape
    
    H_out = H // 2
    W_out = W // 2
    
    out = np.zeros((B, H_out, W_out, C))

    for i in range(H_out):
        for j in range(W_out):
            window = x[:, i*2:i*2+2, j*2:j*2+2, :]   # 2x2 region
            out[:, i, j, :] = np.max(window, axis=(1, 2))

    return out