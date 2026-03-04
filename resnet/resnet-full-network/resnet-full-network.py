import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.downsample = downsample
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        # Projection shortcut if dimensions change (or downsample)
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if (in_ch != out_ch or downsample) else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass:
          2D: (B,C)  -> (B,out_ch)
          4D: (B,C,H,W) -> (B,out_ch,H',W')  (H',W' halved if downsample)
        """
        x = np.asarray(x)

        # --- Handle (C,) as (1,C)
        squeeze_back = False
        if x.ndim == 1:
            x = x[None, :]
            squeeze_back = True

        # --- 2D path: treat as per-sample "conv"
        if x.ndim == 2:
            main = relu(x @ self.W1)
            main = main @ self.W2

            if self.W_proj is None:
                skip = x
            else:
                skip = x @ self.W_proj

            y = relu(main + skip)
            return y[0] if squeeze_back else y

        # --- 4D path: (B,C,H,W)
        if x.ndim == 4:
            x_ds = x[:, :, ::2, ::2] if self.downsample else x  # stride=2 subsample

            # channels-last for linear maps: (B,H,W,C)
            xm = x_ds.transpose(0, 2, 3, 1)

            main = relu(xm @ self.W1)
            main = main @ self.W2

            if self.W_proj is None:
                skip = xm
            else:
                skip = xm @ self.W_proj

            y = relu(main + skip)
            return y.transpose(0, 3, 1, 2)  # back to (B,C,H,W)

        raise ValueError(f"BasicBlock expects 1D, 2D, or 4D input, got shape {x.shape}")

class ResNet18:
    """
    Simplified ResNet-18 architecture.

    Works for:
      - images: (B,3,H,W) or (3,H,W)
      - vectors: (B,3) or (3,)
    Returns:
      logits: (B, num_classes)
    """
    
    def __init__(self, num_classes: int = 10):
        self.conv1 = np.random.randn(3, 64) * 0.01
        
        # 2 blocks per stage (8 total)
        self.layer1 = [BasicBlock(64, 64, downsample=False),
                       BasicBlock(64, 64, downsample=False)]
        
        self.layer2 = [BasicBlock(64, 128, downsample=True),
                       BasicBlock(128, 128, downsample=False)]
        
        self.layer3 = [BasicBlock(128, 256, downsample=True),
                       BasicBlock(256, 256, downsample=False)]
        
        self.layer4 = [BasicBlock(256, 512, downsample=True),
                       BasicBlock(512, 512, downsample=False)]
        
        self.fc = np.random.randn(512, num_classes) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet-18.
        """
        x = np.asarray(x)

        # ---- Normalize input to have batch dim
        was_single = False
        if x.ndim in (1, 3):   # (3,) or (3,H,W)
            x = x[None, ...]
            was_single = True

        # ---- Vector input: (B,3)
        if x.ndim == 2:
            h = relu(x @ self.conv1)  # (B,64)

            for blk in self.layer1: h = blk.forward(h)
            for blk in self.layer2: h = blk.forward(h)
            for blk in self.layer3: h = blk.forward(h)
            for blk in self.layer4: h = blk.forward(h)

            logits = h @ self.fc      # (B,num_classes)
            return logits

        # ---- Image input: (B,3,H,W)
        if x.ndim == 4:
            # conv1: per-pixel linear map 3->64
            xp = x.transpose(0, 2, 3, 1)   # (B,H,W,3)
            xp = relu(xp @ self.conv1)     # (B,H,W,64)
            h = xp.transpose(0, 3, 1, 2)   # (B,64,H,W)

            for blk in self.layer1: h = blk.forward(h)
            for blk in self.layer2: h = blk.forward(h)
            for blk in self.layer3: h = blk.forward(h)
            for blk in self.layer4: h = blk.forward(h)

            # global avg pool: (B,512)
            h = h.mean(axis=(2, 3))
            logits = h @ self.fc          # (B,num_classes)
            return logits

        raise ValueError(f"ResNet18 forward expects 2D or 4D (after batching), got shape {x.shape}")