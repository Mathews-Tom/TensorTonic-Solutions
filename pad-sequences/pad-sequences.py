import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    N = len(seqs)

    if max_len is None:
        L = max((len(seq) for seq in seqs), default=0)
    else:
        L = max_len

    out = np.full((N, L), pad_value)

    for i, seq in enumerate(seqs):
        trunc = seq[:L]
        out[i, :len(trunc)] = trunc

    return out