import torch
import torch.nn as nn
import math

# Input Embedding
class InputEmbedding(nn.Module):
    """Input embedding layer untuk setiap token"""
    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model (int)   : Dimensi model -> dimensi embedding
            vocab_size (int): Ukuran vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

# Positional Encoding