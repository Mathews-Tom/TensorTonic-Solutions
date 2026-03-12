import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)

    if rng is None:
        random_values = np.random.random(x.shape)
    else:
        random_values = rng.random(x.shape)

    scale = 1.0 / (1.0 - p)
    keep_mask = random_values < (1.0 - p)

    dropout_pattern = keep_mask.astype(float) * scale
    output = x * dropout_pattern

    return output, dropout_pattern