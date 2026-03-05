import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.

    Args:
        patches: (B, N, D)
        embed_dim: D

    Returns:
        (B, N+1, D) with [CLS] token at position 0
    """
    if not isinstance(patches, np.ndarray) or patches.ndim != 3:
        raise ValueError("patches must be a 3D numpy array (B, N, D)")

    B, N, D = patches.shape

    if D != embed_dim:
        raise ValueError("embed_dim must match patches.shape[2]")

    # Initialize learnable CLS token (1,1,D)
    rng = np.random.default_rng(42)
    cls_token = rng.normal(0, 0.02, size=(1, 1, D)).astype(np.float32)

    # Tile across batch
    cls_tokens = np.tile(cls_token, (B, 1, 1))  # (B,1,D)

    # Concatenate at beginning of sequence
    output = np.concatenate([cls_tokens, patches], axis=1)

    return output