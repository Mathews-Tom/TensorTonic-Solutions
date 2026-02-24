import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # (m, n)
    X = np.array(X, dtype=float)
    # (m,)
    y = np.array(y, dtype=float).reshape(-1)

    m, n = X.shape
    # (n,)
    w = np.zeros(n, dtype=float)
    b = 0.0

    for _ in range(steps):
        # (m,)
        z = X @ w + b                     
        # (m,)
        p = _sigmoid(z)

        # (m,)
        error = p - y
        # (n,)
        dw = (X.T @ error) / m
        # scalar
        db = np.sum(error) / m

        w -= lr * dw
        b -= lr * db

    return w, b