import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net for segmentation (shape simulation).
    Input:  (B, H, W, C)
    Output: (B, H_out, W_out, num_classes)
    """

    x = np.asarray(x)
    B, H, W, C = x.shape

    def encoder_block(inp, out_ch):
        # two valid 3x3 convs => H,W - 4; channels -> out_ch
        B, H, W, _ = inp.shape
        skip = np.zeros((B, H - 4, W - 4, out_ch))
        # 2x2 maxpool stride 2 => halve spatial
        pool = np.zeros((B, (H - 4) // 2, (W - 4) // 2, out_ch))
        return pool, skip

    def bottleneck(inp, out_ch):
        # two valid 3x3 convs => H,W - 4; channels -> out_ch
        B, H, W, _ = inp.shape
        return np.zeros((B, H - 4, W - 4, out_ch))

    def crop_and_concat(enc, dec):
        # center crop encoder to decoder spatial, concat channels
        B, He, We, Ce = enc.shape
        Bd, Hd, Wd, Cd = dec.shape
        top = (He - Hd) // 2
        left = (We - Wd) // 2
        enc_crop = enc[:, top:top + Hd, left:left + Wd, :]
        return np.zeros((B, Hd, Wd, Ce + Cd))

    def decoder_block(inp, skip, out_ch):
        # upsample spatial x2, set channels -> out_ch (up-conv)
        B, H, W, _ = inp.shape
        up = np.zeros((B, 2 * H, 2 * W, out_ch))

        # crop+concat with skip
        merged = crop_and_concat(skip, up)  # (B, 2H, 2W, out_ch + Cskip)

        # two valid 3x3 convs => spatial -4 total; channels -> out_ch
        _, Hm, Wm, _ = merged.shape
        return np.zeros((B, Hm - 4, Wm - 4, out_ch))

    def output_layer(inp, ncls):
        B, H, W, _ = inp.shape
        return np.zeros((B, H, W, ncls))

    # Classic channels: 64, 128, 256, 512, bottleneck 1024
    p1, s1 = encoder_block(x, 64)
    p2, s2 = encoder_block(p1, 128)
    p3, s3 = encoder_block(p2, 256)
    p4, s4 = encoder_block(p3, 512)

    b = bottleneck(p4, 1024)

    d4 = decoder_block(b,  s4, 512)
    d3 = decoder_block(d4, s3, 256)
    d2 = decoder_block(d3, s2, 128)
    d1 = decoder_block(d2, s1, 64)

    return output_layer(d1, num_classes)