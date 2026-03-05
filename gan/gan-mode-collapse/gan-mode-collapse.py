import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Detect mode collapse in generated samples by measuring how low the
    standard deviation (diversity) of generated outputs is.

    Returns:
        {
          "diversity_score": float,   # normalized so N(0,1) ~ 1.0
          "is_collapsed": bool
        }
    """
    if not isinstance(generated_samples, np.ndarray):
        raise TypeError("generated_samples must be a numpy.ndarray")
    if generated_samples.ndim != 2:
        raise ValueError(f"generated_samples must be 2D (n_samples, dim), got shape {generated_samples.shape}")
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError("threshold must be a non-negative number")

    # Compute per-dimension std across samples, then average -> overall diversity
    per_dim_std = np.std(generated_samples, axis=0)
    diversity = float(np.mean(per_dim_std))

    # Normalize so samples ~ N(0,1) have diversity_score ~ 1.0
    diversity_score = diversity  # since std of N(0,1) is ~1 per dim on average

    is_collapsed = diversity_score < float(threshold)

    return {"diversity_score": float(diversity_score), "is_collapsed": bool(is_collapsed)}