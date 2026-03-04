import numpy as np

# Helper functions are provided by the platform:
# - vgg_features(x, config)
# - vgg_classifier(features, num_classes)

def vgg16(x: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    """
    Implement the complete VGG-16 network:
      x -> vgg_features (VGG16 config) -> vgg_classifier -> logits
    """
    config = [64, 64, "M",
              128, 128, "M",
              256, 256, 256, "M",
              512, 512, 512, "M",
              512, 512, 512, "M"]

    feats = vgg_features(x, config)              # (B, H', W', 512)
    logits = vgg_classifier(feats, num_classes)  # (B, num_classes)
    return logits