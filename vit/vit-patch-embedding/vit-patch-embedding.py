import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.

    Args:
        image: (B, H, W, C)
        patch_size: P (must divide H and W)
        embed_dim: D

    Returns:
        embeddings: (B, N, D) where N = (H/P) * (W/P)
    """
    if not isinstance(image, np.ndarray) or image.ndim != 4:
        raise ValueError(f"image must be a 4D numpy array (B,H,W,C), got {type(image)} with shape {getattr(image,'shape',None)}")
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError("embed_dim must be a positive integer")

    B, H, W, C = image.shape
    P = patch_size

    if H % P != 0 or W % P != 0:
        raise ValueError("patch_size must evenly divide both H and W")

    h_patches = H // P
    w_patches = W // P
    N = h_patches * w_patches
    patch_dim = P * P * C

    # ---- Extract non-overlapping patches ----
    # (B, H, W, C) -> (B, h_patches, P, w_patches, P, C)
    patches = image.reshape(B, h_patches, P, w_patches, P, C)
    # -> (B, h_patches, w_patches, P, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    # -> (B, N, patch_dim)
    patches = patches.reshape(B, N, patch_dim)

    # ---- Linear projection to embedding dim D ----
    # Deterministic weights based on shape (so function is pure and repeatable)
    seed = (patch_dim * 1_000_003 + embed_dim) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    W_proj = rng.standard_normal((patch_dim, embed_dim), dtype=np.float32) * np.sqrt(2.0 / (patch_dim + embed_dim))
    b_proj = np.zeros((embed_dim,), dtype=np.float32)

    patches = patches.astype(np.float32, copy=False)
    embeddings = patches @ W_proj + b_proj  # (B, N, D)

    return embeddings