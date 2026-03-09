import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute cumulative product of (1 - beta).
    """
    alphas = 1.0 - betas
    return np.cumprod(alphas)

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    betas: np.ndarray
) -> tuple:
    """
    Sample x_t from q(x_t | x_0).
    """
    alpha_bar = get_alpha_bar(betas)
    
    # t is 1-indexed, so use t-1 for array indexing
    alpha_bar_t = alpha_bar[t - 1]
    
    epsilon = np.random.randn(*x_0.shape)
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1.0 - alpha_bar_t) * epsilon
    
    return x_t, epsilon