import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    Returns array of shape (seq_length, d_model)
    """
    # Positions: (seq_length, 1)
    positions = np.arange(seq_length)[:, np.newaxis]

    # Dimension indices: (1, d_model)
    dims = np.arange(d_model)[np.newaxis, :]

    # Compute the angle rates
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)

    # Compute angle matrix
    angles = positions * angle_rates

    # Initialize encoding matrix
    pe = np.zeros((seq_length, d_model))

    # Apply sin to even indices, cos to odd indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe