import numpy as np
from typing import List

class MockBertEncoder:
    """Simulated BERT encoder with 12 layers."""
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Each layer just adds a small transformation
        self.layers = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_frozen = [False] * num_layers
    
    def freeze_layers(self, layer_indices: List[int]):
        """Freeze specified layers (no gradient updates)."""
        for idx in layer_indices:
            if 0 <= idx < self.num_layers:
                self.layer_frozen[idx] = True
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        self.layer_frozen = [False] * self.num_layers
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        x = embeddings
        for i, layer in enumerate(self.layers):
            # In this mock, frozen affects training updates, not forward computation.
            x = x @ layer + x  # Simplified residual
        return x

class BertForSequenceClassification:
    """BERT with classification head."""
    
    def __init__(self, hidden_size: int, num_labels: int, freeze_bert: bool = False):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
        self.freeze_bert = freeze_bert
        
        if freeze_bert:
            self.encoder.freeze_layers(list(range(self.encoder.num_layers)))
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass for sequence classification.
        embeddings: (batch, seq_len, hidden_size)
        returns logits: (batch, num_labels)
        """
        hidden_states = self.encoder.forward(embeddings)          # (B, S, H)
        cls_hidden = hidden_states[:, 0, :]                       # (B, H)
        logits = cls_hidden @ self.classifier                     # (B, num_labels)
        return logits

class BertForTokenClassification:
    """BERT with token-level classification (NER, POS tagging)."""
    
    def __init__(self, hidden_size: int, num_labels: int):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass for token classification.
        embeddings: (batch, seq_len, hidden_size)
        returns logits: (batch, seq_len, num_labels)
        """
        hidden_states = self.encoder.forward(embeddings)          # (B, S, H)
        logits = hidden_states @ self.classifier                  # (B, S, num_labels)
        return logits