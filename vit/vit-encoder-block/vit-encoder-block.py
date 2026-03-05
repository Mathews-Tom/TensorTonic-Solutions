import numpy as np

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> np.ndarray:
    """
    ViT Transformer encoder block (Pre-LN).
    x'  = x  + MSA(LN(x))
    x'' = x' + MLP(LN(x'))   where MLP: D -> (mlp_ratio*D) -> D with GELU

    Args:
        x: (B, N, D)
        embed_dim: D
        num_heads: number of attention heads (must divide D)
        mlp_ratio: hidden dim multiplier for MLP

    Returns:
        (B, N, D)
    """
    if not isinstance(x, np.ndarray) or x.ndim != 3:
        raise ValueError("x must be a 3D numpy array (B, N, D)")
    B, N, D = x.shape
    if D != embed_dim:
        raise ValueError("embed_dim must match x.shape[2]")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    if D % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0:
        raise ValueError("mlp_ratio must be positive")

    x = x.astype(np.float32, copy=False)

    # ---------------- helpers ----------------
    def layer_norm(t: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = t.mean(axis=-1, keepdims=True)
        var = ((t - mean) ** 2).mean(axis=-1, keepdims=True)
        return (t - mean) / np.sqrt(var + eps)

    def gelu(t: np.ndarray) -> np.ndarray:
        # GELU approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
        return 0.5 * t * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (t + 0.044715 * (t ** 3))))

    def softmax(t: np.ndarray, axis: int = -1) -> np.ndarray:
        t = t - np.max(t, axis=axis, keepdims=True)
        exp_t = np.exp(t)
        return exp_t / np.sum(exp_t, axis=axis, keepdims=True)

    # Deterministic weights (pure function; no state)
    seed = (D * 1_000_003 + N * 9176 + num_heads * 101) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    # ---------------- MSA(LN(x)) ----------------
    x_ln = layer_norm(x)

    head_dim = D // num_heads

    # Linear projections: (B,N,D) -> (B,N,D) for q,k,v
    # Use separate matrices for Q,K,V and output projection
    scale_qkv = np.sqrt(2.0 / (D + D))
    Wq = rng.standard_normal((D, D), dtype=np.float32) * scale_qkv
    Wk = rng.standard_normal((D, D), dtype=np.float32) * scale_qkv
    Wv = rng.standard_normal((D, D), dtype=np.float32) * scale_qkv
    bq = np.zeros((D,), dtype=np.float32)
    bk = np.zeros((D,), dtype=np.float32)
    bv = np.zeros((D,), dtype=np.float32)

    Wo = rng.standard_normal((D, D), dtype=np.float32) * scale_qkv
    bo = np.zeros((D,), dtype=np.float32)

    q = x_ln @ Wq + bq
    k = x_ln @ Wk + bk
    v = x_ln @ Wv + bv

    # Reshape to heads: (B,N,D) -> (B,H,N,hd)
    q = q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention: scores = q k^T / sqrt(hd)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(float(head_dim))  # (B,H,N,N)
    attn = softmax(scores, axis=-1)
    context = attn @ v  # (B,H,N,hd)

    # Merge heads: (B,H,N,hd) -> (B,N,D)
    context = context.transpose(0, 2, 1, 3).reshape(B, N, D)

    msa_out = context @ Wo + bo
    x1 = x + msa_out  # residual

    # ---------------- MLP(LN(x1)) ----------------
    x1_ln = layer_norm(x1)
    hidden_dim = int(round(mlp_ratio * D))

    scale_mlp1 = np.sqrt(2.0 / (D + hidden_dim))
    scale_mlp2 = np.sqrt(2.0 / (hidden_dim + D))
    W1 = rng.standard_normal((D, hidden_dim), dtype=np.float32) * scale_mlp1
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    W2 = rng.standard_normal((hidden_dim, D), dtype=np.float32) * scale_mlp2
    b2 = np.zeros((D,), dtype=np.float32)

    h = gelu(x1_ln @ W1 + b1)
    mlp_out = h @ W2 + b2

    x2 = x1 + mlp_out  # residual
    return x2