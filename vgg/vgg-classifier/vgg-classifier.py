import numpy as np

def vgg_classifier(features: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    """
    Implement VGG's fully connected classifier.
    
    Input:  features (B, 7, 7, 512)
    Output: logits (B, num_classes)
    """
    
    B = features.shape[0]
    
    # Flatten
    x = features.reshape(B, -1)  # (B, 25088)

    # FC1
    W1 = np.random.randn(x.shape[1], 4096) * 0.01
    b1 = np.zeros(4096)
    x = x @ W1 + b1
    x = np.maximum(0, x)  # ReLU

    # FC2
    W2 = np.random.randn(4096, 4096) * 0.01
    b2 = np.zeros(4096)
    x = x @ W2 + b2
    x = np.maximum(0, x)  # ReLU

    # FC3 (classifier)
    W3 = np.random.randn(4096, num_classes) * 0.01
    b3 = np.zeros(num_classes)
    logits = x @ W3 + b3

    return logits