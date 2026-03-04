import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    For a residual block: y = x + F(x), the local Jacobian is (I + dF/dx).
    Overall gradient: (Π_l (I + J_l)) x
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    g = x

    for J in gradients_F:
        J = np.asarray(J, dtype=float)

        # Support scalar "Jacobian" (1D case)
        if J.ndim == 0:
            g = (1.0 + J) * g
            continue

        I = np.eye(J.shape[0], dtype=float)
        g = (I + J) @ g

    return g


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    Overall gradient: (Π_l J_l) x
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    g = x

    for J in gradients_F:
        J = np.asarray(J, dtype=float)

        # Support scalar "Jacobian" (1D case)
        if J.ndim == 0:
            g = J * g
            continue

        g = J @ g

    return g