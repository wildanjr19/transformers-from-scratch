"""
Script untuk mendefinisikan model transformers
    class Transformers: mendefinisikan arsitektur model transformers
    build_transformers: fungsi untuk membangun dan menginisialisasi model transformers
"""

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
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # (bh, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    # projection
    def projection(self, x):
        return self.projection_layer(x)
    
# fungsi untuk membangun dan menginisalisasi model transformers
def build_transformers(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformers:
    """
    Args:
        src_vocab_size = vocab size data source
        tgt_vocab_size = vocab size data target
        src_seq_len = panjang kalimat data source
        tgt_seq_len = panjang kalimat data target
        d_model = dimensi embedding
        N = jumlah layer pada encoder dan decoder block
        h =  banyak kepala MHA
        d_ff = dimensi feed forward
    """
    # buat embedding layer untuk masing-masing input
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # baut positional encoding layer
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # buat encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # buat decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # buat encoder dan decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    # projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) 

    # buat instance transformers
    transformer = Transformers(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # inisialisasi weight parameter dengan distribusi uniform untuk stabilitas
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer