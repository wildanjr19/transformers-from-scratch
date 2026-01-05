"""
Script untuk mendefinisikan Decoder dan DecoderBlock
    class DecoderBlock: mendefinisikan satu blok decoder
    class Decoder: mendefinisikan tumpukan beberapa DecoderBlock
"""

import torch
import torch.nn as nn
from modules import MultiHeadAttention, FeedForwardBlock, ResidualConnection, LayerNormalization

class DecoderBlock(nn.Module): 
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        """
        DecoderBlock -> Bagian dari Decoder, yang terdiri dari beberapa layer (self-attention, cross-attention, feed-forward) dengan residual connection dan layer normalization

        Args:
            self_attention_block : Multi head attention layer
            cross_attention_block : Multi head attention layer untuk cross attention
            feed_forward_block : FFN layer
        """
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Add & Norm (3x)
        self.add_norm = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # fase 1
        x = self.add_norm[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # fase 2 (k dan v dari encoder, q dari input)
        x = self.add_norm[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # fase 3
        x = self.add_norm[2](x, lambda x: self.feed_forward_block(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        """
        Decoder -> tumpukan beberapa DecoderBlock

        Args:
            features (int) : fitur
            layers (Module List) : daftar lapisan DecoderBlock
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)