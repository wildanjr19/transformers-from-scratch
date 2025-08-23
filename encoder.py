import torch
import torch.nn as nn
from modules import MultiHeadAttention, FeedForwardBlock, ResidualConnection, LayerNormalization

class EncoderBlock(nn.Module):
    """Encoder Block"""
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        """
        Args:
            self_attention_block : Multi head attention layer
            feed_forward_block : FFN layer
        """
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Add & Norm (2x)
        self.add_norm = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # fase 1
        x = self.add_norm[0](x, lambda x: self.self_attention_block(x,x, x, src_mask))
        # fase 2
        x = self.add_norm[1](x, lambda x: self.feed_forward_block(x))

        return x


class Encoder(nn.Module):
    """Encoder -> tumpukan beberapa EncoderBlock"""
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        """
        Args:
            features (int) : fitur
            layers (Module List) : daftar lapisan EncoderBlock
        """
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)