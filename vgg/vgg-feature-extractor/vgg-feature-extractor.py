import numpy as np

def conv_relu(x, out_channels):
    B, H, W, C = x.shape
    W_weights = np.random.randn(C, out_channels) * 0.1
    x = x @ W_weights
    return np.maximum(0, x)

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2,4))

def vgg_features(x: np.ndarray, config: list) -> np.ndarray:
    """
    Build VGG feature extractor from config.
    Integers => conv_relu to that many channels
    'M' => 2x2 maxpool with stride 2
    """
    out = x
    for layer in config:
        if layer == "M":
            out = maxpool_2x2(out)
        else:
            out = conv_relu(out, int(layer))
    return out