import numpy as np
import math

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization over the last dimension.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    Shapes:
      Q, K, V: (batch, seq_len, d_model)
      W_q, W_k, W_v, W_o: (d_model, d_model)
    Returns:
      (batch, seq_len, d_model)
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    V = np.asarray(V)

    b, seq_q, d_model = Q.shape
    _, seq_k, d_model_k = K.shape
    _, seq_v, d_model_v = V.shape

    if not (d_model == d_model_k == d_model_v):
        raise ValueError("Q, K, V must have same d_model.")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")

    d_k = d_model // num_heads

    # Linear projections
    Qp = Q @ W_q
    Kp = K @ W_k
    Vp = V @ W_v

    # Split heads: (b, h, seq, d_k)
    def split_heads(x):
        return x.reshape(b, -1, num_heads, d_k).transpose(0, 2, 1, 3)

    Qh = split_heads(Qp)  # (b,h,seq_q,d_k)
    Kh = split_heads(Kp)  # (b,h,seq_k,d_k)
    Vh = split_heads(Vp)  # (b,h,seq_v,d_k)

    # Attention: scores (b,h,seq_q,seq_k)
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    heads_out = weights @ Vh  # (b,h,seq_q,d_k)

    # Concat heads -> (b,seq_q,d_model)
    concat = heads_out.transpose(0, 2, 1, 3).reshape(b, seq_q, d_model)

    # Output projection
    return concat @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    Shapes:
      x:  (batch, seq_len, d_model)
      W1: (d_model, d_ff), b1: (d_ff,)
      W2: (d_ff, d_model), b2: (d_model,)
    """
    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)  # ReLU
    return hidden @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block (Post-LN):
      x1 = LayerNorm(x + MHA(x,x,x))
      out = LayerNorm(x1 + FFN(x1))
    """
    # Self-attention + residual
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x1 = layer_norm(x + attn_out, gamma1, beta1)

    # FFN + residual
    ffn_out = feed_forward(x1, W1, b1, W2, b2)
    out = layer_norm(x1 + ffn_out, gamma2, beta2)

    return out