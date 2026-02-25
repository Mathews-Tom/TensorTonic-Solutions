import numpy as np
import math

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    Shapes:
      Q, K, V: (batch, seq_len, d_model)
      W_q, W_k, W_v: (d_model, d_model)
      W_o: (d_model, d_model)
    Returns:
      (batch, seq_len, d_model)
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    V = np.asarray(V)

    batch, seq_len_q, d_model = Q.shape
    _, seq_len_k, d_model_k = K.shape
    _, seq_len_v, d_model_v = V.shape
    if not (d_model == d_model_k == d_model_v):
        raise ValueError("Q, K, V must have the same d_model in the last dimension.")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")

    d_k = d_model // num_heads

    # Linear projections: (batch, seq_len, d_model)
    Qp = Q @ W_q
    Kp = K @ W_k
    Vp = V @ W_v

    # Split into heads: (batch, num_heads, seq_len, d_k)
    def split_heads(x):
        return x.reshape(batch, -1, num_heads, d_k).transpose(0, 2, 1, 3)

    Qh = split_heads(Qp)  # (b, h, sq, dk)
    Kh = split_heads(Kp)  # (b, h, sk, dk)
    Vh = split_heads(Vp)  # (b, h, sv, dk)

    # Scaled dot-product attention per head:
    # scores: (b, h, sq, sk)
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
    weights = softmax(scores, axis=-1)  # (b, h, sq, sk)

    # head outputs: (b, h, sq, dk)
    heads_out = weights @ Vh

    # Concatenate heads: (b, sq, d_model)
    concat = heads_out.transpose(0, 2, 1, 3).reshape(batch, seq_len_q, d_model)

    # Final linear projection: (b, sq, d_model)
    out = concat @ W_o
    return out