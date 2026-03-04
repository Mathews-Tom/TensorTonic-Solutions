import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply BERT's MLM masking strategy (80% [MASK], 10% random, 10% unchanged).

    Returns:
        masked_token_ids: same shape as token_ids
        labels: same shape as token_ids, -100 for non-masked positions, original token id for masked
        mask_positions: boolean array same shape as token_ids (True where masked)
    """
    if seed is not None:
        np.random.seed(seed)

    token_ids = np.asarray(token_ids, dtype=int)
    masked_token_ids = token_ids.copy()

    # Choose mask positions
    rand = np.random.rand(*token_ids.shape)
    mask_positions = rand < mask_prob

    # Labels: only masked positions contribute to loss
    labels = np.full(token_ids.shape, -100, dtype=int)
    labels[mask_positions] = token_ids[mask_positions]

    # For masked positions, decide 80/10/10 behavior
    choices = np.random.rand(*token_ids.shape)

    # 80% -> [MASK]
    mask_replace = mask_positions & (choices < 0.8)
    masked_token_ids[mask_replace] = mask_token_id

    # 10% -> random token
    random_replace = mask_positions & (choices >= 0.8) & (choices < 0.9)
    if np.any(random_replace):
        masked_token_ids[random_replace] = np.random.randint(0, vocab_size, size=np.sum(random_replace))

    # 10% -> unchanged (do nothing)

    return masked_token_ids, labels, mask_positions


class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits.

        Args:
            hidden_states: (batch, seq_len, hidden_size) or (seq_len, hidden_size)

        Returns:
            logits: (batch, seq_len, vocab_size) or (seq_len, vocab_size)
        """
        hs = np.asarray(hidden_states, dtype=float)

        if hs.ndim == 2:
            # (seq_len, hidden_size)
            return hs @ self.W + self.b

        if hs.ndim == 3:
            # (batch, seq_len, hidden_size)
            return hs @ self.W + self.b  # broadcasts over first dims

        raise ValueError(f"hidden_states must be 2D or 3D, got shape {hs.shape}")