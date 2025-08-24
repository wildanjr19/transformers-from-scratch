import torch
import torch.nn as nn
from modules import *
from encoder import *
from decoder import *

class Transformers(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder      # encoder block
        self.decoder = decoder      # decoder block
        self.src_embed = src_embed  # embedding dari encoder
        self.tgt_embed = tgt_embed  # embedding dari decoder
        self.src_pos = src_pos      # positional encoding dari encoder
        self.tgt_pos = tgt_pos      # positional encoding dari decoder
        self.projection_layer = projection_layer  # projection layer

    # encoding
    def encode(self, x, src_mask):
        # (bh, seq_len, d_model)
        x = self.src_embed(x)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)
    
    # decoding
    def decode(self, tgt, encoder_output,  src_mask, tgt_mask):
        # (bh, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    # projection
    def projection(self, x):
        return self.projection_layer(x)