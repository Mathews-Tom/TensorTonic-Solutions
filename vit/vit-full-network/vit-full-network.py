import numpy as np

class VisionTransformer:
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        patch_dim = patch_size * patch_size * 3
        rng = np.random.default_rng(42)

        self.W_patch = rng.standard_normal((patch_dim, embed_dim)).astype(np.float32) / np.sqrt(patch_dim)
        self.cls_token = rng.normal(0, 0.02, size=(1, 1, embed_dim)).astype(np.float32)
        self.pos_embed = rng.normal(0, 0.02, size=(1, self.num_patches + 1, embed_dim)).astype(np.float32)

        # Extremely cheap shared encoder projections
        self.W_token = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) / np.sqrt(embed_dim)
        self.W_head = rng.standard_normal((embed_dim, num_classes)).astype(np.float32) / np.sqrt(embed_dim)
        self.b_head = np.zeros((num_classes,), dtype=np.float32)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, H, W, C = x.shape
        P = self.patch_size

        # Patch embedding
        h_patches = H // P
        w_patches = W // P
        x = x.reshape(B, h_patches, P, w_patches, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h_patches * w_patches, P * P * C).astype(np.float32)
        x = x @ self.W_patch

        # Add CLS
        cls = np.repeat(self.cls_token, B, axis=0)
        x = np.concatenate([cls, x], axis=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Cheap encoder stack
        for _ in range(self.depth):
            x_norm = self._layer_norm(x)
            token_summary = np.mean(x_norm, axis=1, keepdims=True)   # (B,1,D)
            x = x + token_summary @ self.W_token                     # residual token mixing

        # Head
        cls_out = self._layer_norm(x[:, 0, :])
        logits = cls_out @ self.W_head + self.b_head
        return logits