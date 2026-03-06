import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    
    ||∇h_0|| ≈ ||W_hh||_2^T
    
    We simulate the gradient magnitude as it propagates
    backward through time.

    Returns:
        List of gradient norms for each step (length T),
        normalized so the first value = 1.0
    """
    # spectral norm (largest singular value)
    spectral_norm = np.linalg.norm(W_hh, ord=2)

    grad = 1.0
    norms = [grad]

    for _ in range(1, T):
        grad *= spectral_norm
        norms.append(grad)

    return norms