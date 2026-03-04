import numpy as np
from typing import List, Tuple
import random

def create_nsp_examples(
    documents: List[List[str]], 
    num_examples: int,
    seed: int = None
) -> List[Tuple[str, str, int]]:
    """
    Create NSP training examples.
    Returns list of (sentence_a, sentence_b, is_next) where:
      is_next = 1 for true next sentence pair
      is_next = 0 for random (not next) pair
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Collect all sentences for negative sampling
    all_sents: List[Tuple[int, int, str]] = []
    for di, doc in enumerate(documents):
        for si, s in enumerate(doc):
            all_sents.append((di, si, s))

    examples: List[Tuple[str, str, int]] = []
    if len(all_sents) == 0:
        return examples

    while len(examples) < num_examples:
        make_is_next = (random.random() < 0.5)

        if make_is_next:
            # Pick a doc with at least 2 sentences, then pick a consecutive pair
            valid_docs = [d for d in documents if len(d) >= 2]
            if not valid_docs:
                # Fallback: if no doc has 2 sentences, can't make positives
                make_is_next = False
            else:
                doc = random.choice(valid_docs)
                i = random.randrange(0, len(doc) - 1)
                sent_a = doc[i]
                sent_b = doc[i + 1]
                examples.append((sent_a, sent_b, 1))
                continue

        # NotNext (random) pair
        # Choose A from anywhere
        doc_a_idx, sent_a_idx, sent_a = random.choice(all_sents)

        # Choose B randomly, but ensure it's NOT the true next sentence of A
        while True:
            doc_b_idx, sent_b_idx, sent_b = random.choice(all_sents)

            # Reject if it's the true next sentence in the same doc
            if doc_b_idx == doc_a_idx and sent_b_idx == sent_a_idx + 1:
                continue
            # Also reject identical sentence position (optional but avoids trivial duplicates)
            if doc_b_idx == doc_a_idx and sent_b_idx == sent_a_idx:
                continue
            break

        examples.append((sent_a, sent_b, 0))

    return examples


class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict NSP logits from [CLS] hidden state.

        Args:
            cls_hidden: (hidden_size,) or (batch, hidden_size)

        Returns:
            logits: (2,) or (batch, 2)
        """
        x = np.asarray(cls_hidden, dtype=float)

        if x.ndim == 1:
            return x @ self.W + self.b
        if x.ndim == 2:
            return x @ self.W + self.b

        raise ValueError(f"cls_hidden must be 1D or 2D, got shape {x.shape}")


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)