import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Classification head for ViT.

    Args:
        encoder_output: (B, N, D) transformer output
        num_classes: number of classes

    Returns:
        logits: (B, num_classes)
    """
    if not isinstance(encoder_output, np.ndarray) or encoder_output.ndim != 3:
        raise ValueError("encoder_output must be a 3D numpy array (B, N, D)")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")

    B, N, D = encoder_output.shape

    # ---- Extract CLS token ----
    cls_token = encoder_output[:, 0, :]  # (B, D)

    # ---- Layer Normalization ----
    eps = 1e-5
    mean = cls_token.mean(axis=-1, keepdims=True)
    var = ((cls_token - mean) ** 2).mean(axis=-1, keepdims=True)
    cls_norm = (cls_token - mean) / np.sqrt(var + eps)

    # ---- Linear projection to class logits ----
    rng = np.random.default_rng(42)
    W = rng.standard_normal((D, num_classes), dtype=np.float32) * np.sqrt(2.0 / (D + num_classes))
    b = np.zeros((num_classes,), dtype=np.float32)

    logits = cls_norm.astype(np.float32) @ W + b  # (B, num_classes)

    return logits