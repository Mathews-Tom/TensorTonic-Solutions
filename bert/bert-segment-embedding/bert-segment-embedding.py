import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Compute BERT embeddings.

        Args:
            token_ids: (batch, seq_len)
            segment_ids: (batch, seq_len)

        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings
        token_embed = self.token_embeddings[token_ids]

        # Position embeddings
        position_ids = np.arange(seq_len)
        position_embed = self.position_embeddings[position_ids]
        position_embed = np.broadcast_to(position_embed, (batch_size, seq_len, self.hidden_size))

        # Segment embeddings
        segment_embed = self.segment_embeddings[segment_ids]

        # Sum all components
        embeddings = token_embed + position_embed + segment_embed

        return embeddings