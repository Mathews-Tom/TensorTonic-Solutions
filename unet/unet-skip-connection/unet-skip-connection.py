import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Center-crop encoder_features to match decoder_features spatial size,
    then concatenate along the channel axis.

    Shapes (NHWC):
      encoder_features: (B, H_enc, W_enc, C_enc)
      decoder_features: (B, H_dec, W_dec, C_dec)
    Output:
      (B, H_dec, W_dec, C_enc + C_dec)
    """
    B, H_enc, W_enc, C_enc = encoder_features.shape
    B2, H_dec, W_dec, C_dec = decoder_features.shape
    if B != B2:
        raise ValueError("Batch sizes must match")

    # Center crop indices
    dh = H_enc - H_dec
    dw = W_enc - W_dec
    if dh < 0 or dw < 0:
        raise ValueError("Encoder features must be >= decoder features in spatial dims")

    top = dh // 2
    left = dw // 2

    enc_cropped = encoder_features[:, top:top + H_dec, left:left + W_dec, :]
    return np.concatenate([enc_cropped, decoder_features], axis=-1)