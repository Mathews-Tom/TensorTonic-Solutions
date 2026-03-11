import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # Step 1: update running squared gradient average
    s = beta * s + (1 - beta) * (g ** 2)

    # Step 2: update parameters
    w = w - (lr / (np.sqrt(s) + eps)) * g

    return w, s