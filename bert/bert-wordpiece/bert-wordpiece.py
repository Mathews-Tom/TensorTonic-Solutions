from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT (greedy longest-match-first).
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for word in text.lower().split():
            tokens.extend(self._tokenize_word(word))
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords using greedy longest-match-first.
        Continuation subwords are prefixed with '##'.
        If the word cannot be fully tokenized, return [UNK].
        """
        if len(word) > self.max_word_len:
            return [self.unk_token]

        sub_tokens: List[str] = []
        start = 0
        n = len(word)

        while start < n:
            end = n
            cur_substr = None

            # Find the longest substring in vocab
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece

                if piece in self.vocab:
                    cur_substr = piece
                    break
                end -= 1

            # No matching subword => whole word is unknown
            if cur_substr is None:
                return [self.unk_token]

            sub_tokens.append(cur_substr)
            start = end  # move forward

        return sub_tokens