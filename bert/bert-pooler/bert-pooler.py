import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Pool the [CLS] token representation.
        """
        # hidden_states: (batch, seq_len, hidden_size)

        cls_hidden = hidden_states[:, 0, :]  # extract CLS token
        pooled = cls_hidden @ self.W + self.b
        pooled = tanh(pooled)

        return pooled


class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """
    
    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
    
    def forward(self, hidden_states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Classify sequences.
        """
        pooled = self.pooler.forward(hidden_states)

        if training:
            mask = (np.random.rand(*pooled.shape) > self.dropout_prob).astype(float)
            pooled = pooled * mask / (1.0 - self.dropout_prob)

        logits = pooled @ self.classifier

        return logits