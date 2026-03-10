import numpy as np

def reverse_step(
    x_t: np.ndarray,
    t: int,
    epsilon_pred: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Perform one reverse diffusion step."""
    alpha_t = 1.0 - betas[t]
    alpha_bar_t = np.prod(1.0 - betas[: t + 1])

    # Posterior mean
    mu = (1.0 / np.sqrt(alpha_t)) * (
        x_t - (1.0 - alpha_t) / np.sqrt(1.0 - alpha_bar_t) * epsilon_pred
    )

    if t > 1:
        sigma_t = np.sqrt(betas[t])
        z = np.random.randn(*x_t.shape)
        return mu + sigma_t * z

    return mu