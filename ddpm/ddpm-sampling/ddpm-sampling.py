import numpy as np

def ddpm_sample(
    model_predict: callable,
    shape: tuple,
    betas: np.ndarray,
    T: int
) -> np.ndarray:
    """
    Generate a sample using DDPM.
    """
    x_t = np.random.randn(*shape)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    for t in range(T - 1, -1, -1):
        epsilon_pred = model_predict(x_t, t)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        mu = (1.0 / np.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / np.sqrt(1.0 - alpha_bar_t)) * epsilon_pred
        )

        if t > 1:
            sigma_t = np.sqrt(betas[t])
            z = np.random.randn(*x_t.shape)
            x_t = mu + sigma_t * z
        else:
            x_t = mu

    return x_t