import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.array(X)
    H, W = X.shape
    p, s = pool_size, stride

    H_out = (H - p) // s + 1
    W_out = (W - p) // s + 1

    # Use stride_tricks to extract all pooling windows without loops
    shape = (H_out, W_out, p, p)
    strides = (X.strides[0] * s, X.strides[1] * s, X.strides[0], X.strides[1])
    windows = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

    return windows.max(axis=(2, 3)).tolist()