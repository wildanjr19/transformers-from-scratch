import torch
import torch.nn as nn
import math

# -- INPUT EMBEDDING -- #
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
    

# -- POSITIONAL ENCODING -- #
class PositionalEncoding(nn.Module):
    """Positional Encoding layer"""
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Args:
            d_model (int)   : Dimensi model -> dimensi embedding
            seq_len (int)   : Panjang kalimat 
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # buat matriks 0 dengan ukuran (seq_len, d_model) -> seperti matriks embedding
        pe = torch.zeros(seq_len, d_model)

        # buat vektor dengan ukuran (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sin cos
        pe[:, 0::2] = torch.sin(position * denominator) # indeks genap
        pe[:, 1::2] = torch.cos(position * denominator) # indeks ganjil

        pe = pe.unsqueeze(0) # tambahkan dimensi batch -> (1, seq_len, d_model)

        # register buffer untuk nilai pe
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad(False)
        return self.dropout(x)
    

## -- LAYER NORMALIZATION -- ##
class LayerNormalization(nn.Module):
    """Layer Normalization"""
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # trainable; multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # trainable; added

    def forward(self, x):
        # cari mean dan std di d_model
        mean = x.mean(dim = -1, keepdim= True) #
        std = x.std(dim = - 1, keepdim = True)
        return self.alpha * (x - mean) / (std * self.eps) + self.bias
    

## -- Multi Head Attention -- ## 
class MultiHeadAttention(nn.Module):
    """MHA Block"""
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # ukuran embedding
        self.h = h # banyaknya kepala/head
        # d_model harus divisible oleh h
        assert d_model % h == 0, "d_model harus divisible oleh h"