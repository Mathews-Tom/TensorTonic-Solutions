import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.

    Args:
        patches: (B, N, D)
        num_patches: N
        embed_dim: D

    Returns:
        (B, N, D) with position embeddings added
    """
    if not isinstance(patches, np.ndarray) or patches.ndim != 3:
        raise ValueError("patches must be a 3D numpy array (B, N, D)")

    B, N, D = patches.shape

    if N != num_patches:
        raise ValueError("num_patches must match patches.shape[1]")
    if D != embed_dim:
        raise ValueError("embed_dim must match patches.shape[2]")

    # Initialize learnable position embeddings (N, D)
    rng = np.random.default_rng(42)
    pos_embed = rng.normal(0, 0.02, size=(N, D)).astype(np.float32)

    # Reshape to (1, N, D) for broadcasting
    pos_embed = pos_embed.reshape(1, N, D)

    # Add element-wise
    output = patches + pos_embed

    return output