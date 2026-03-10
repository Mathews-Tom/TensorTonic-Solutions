import numpy as np

def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> np.ndarray:
    """
    Linear noise schedule from beta_1 to beta_T.
    """
    return np.linspace(beta_1, beta_T, T)

def cosine_alpha_bar_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    """
    t = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos(((t + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2
    alpha_bars = f / f[0]
    return alpha_bars[1:]

def alpha_bar_to_betas(alpha_bars: np.ndarray) -> np.ndarray:
    """
    Convert alpha_bar schedule to beta schedule.
    """
    alpha_bars = np.asarray(alpha_bars, dtype=np.float64)
    prev_alpha_bars = np.concatenate(([1.0], alpha_bars[:-1]))
    alphas = alpha_bars / prev_alpha_bars
    betas = 1.0 - alphas
    return np.clip(betas, 1e-8, 0.999)