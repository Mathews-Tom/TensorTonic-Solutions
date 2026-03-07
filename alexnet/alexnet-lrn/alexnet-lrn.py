import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                 alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """Apply Local Response Normalization across channels."""
    
    N, H, W, C = x.shape
    half = n // 2
    
    sq = x ** 2
    scale = np.zeros_like(x)
    
    for i in range(C):
        start = max(0, i - half)
        end = min(C, i + half + 1)
        
        scale[:, :, :, i] = k + alpha * np.sum(sq[:, :, :, start:end], axis=3)
    
    return x / (scale ** beta)