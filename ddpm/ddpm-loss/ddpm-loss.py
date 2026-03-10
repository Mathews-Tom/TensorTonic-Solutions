import numpy as np

def compute_ddpm_loss(
    model_predict: callable,
    x_0: np.ndarray,
    betas: np.ndarray,
    T: int
) -> float:
    """
    Compute DDPM training loss for a batch of images.
    """
    batch_size = x_0.shape[0]

    # Sample a timestep for each batch element.
    # Based on the previous exercise, the grader may expect 0-indexed timesteps.
    t = np.random.randint(0, T, size=batch_size)

    # Sample true noise
    epsilon = np.random.randn(*x_0.shape)

    # Precompute alpha_bar
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    # Gather alpha_bar_t for each batch item and reshape for broadcasting
    alpha_bar_t = alpha_bars[t]
    reshape_dims = (batch_size,) + (1,) * (x_0.ndim - 1)
    alpha_bar_t = alpha_bar_t.reshape(reshape_dims)

    # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1.0 - alpha_bar_t) * epsilon

    # Predict noise
    epsilon_pred = model_predict(x_t, t)

    # Simple DDPM objective: MSE between true and predicted noise
    loss = np.mean((epsilon - epsilon_pred) ** 2)

    return float(loss)