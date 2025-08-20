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

        self.d_k = d_model // h # ukuran setiap head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # weight query
        self.w_k = nn.Linear(d_model, d_model, bias=False) # weight key
        self.w_v = nn.Linear(d_model, d_model, bias=False) # weight value
        self.fc = nn.Linear(d_model, d_model, bias=False) # output

        self.dropout = nn.Dropout(dropout) # dropout

    # attention
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """Self Attention mechanism"""
        # ambil dimensi terakhir sebagai d_k
        d_k = query.shape[-1]

        # scaled dot-product attention : Q @ K^T / SQRT(d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # shape : (bh, h, seq_len, seq_len)
        
        # check mask
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # isi 0 dengan -inf
        
        # apply softmax 
        attention_scores = attention_scores.softmax(dim = -1) 
        
        # dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores) 
            
        # (bh, h, seq_len, seq_len) -> (bh, h, seq_len, d_k), attention_scores
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        # masuk ke masing-masing weight
        query = self.w_q(q)     # (bh, seq_len, d_model) -> (bh, seq_len, d_model)
        key = self.w_k(k)       # (bh, seq_len, d_model) -> (bh, seq_len, d_model)
        value = self.w_v(v)     # (bh, seq_len, d_model) -> (bh, seq_len, d_model)

        # view dan transpose
        # (bh, seq_len, d_model) -> (bh, seq_len, h, d_k) -> (bh, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # hitung attention
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value,  mask, self.dropout)

        # concat
        # (bh, h, seq_len, d_k) -> (bh, seq_len, h, d_k) -> (bh, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # masukkan ke fc
        x = self.fc(x)
        return x
    

## -- RESIDUAL CONNECTION -- ##
class ResidualConnection(nn.Module):
    """ResNet dengan LayerNorm"""
    def __init__(self, dropout: float, features: int) -> None:
        """
        Args:
            dropout (float): dropout
            features (int): jumlah fitur input.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self,  x, sublayer):
        """
        Args:
            sublayer: Sublayer function.
        """
        # return x + self.dropout(sublayer(self.norm(x))) # pre-norm
        return self.norm(x + self.dropout(sublayer(x)))  # residual connection + layer normalization