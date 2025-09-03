# Transformers From Scratch

Implementasi Transformer dari awal menggunakan PyTorch untuk tugas terjemahan bahasa (English-Indonesian).

## Struktur File

- `SelfAttention.py` Implementasi Self Attention Mechanism yang dipakai di Transformers
- `modules.py` Berisi layer-layer pada model (InputEmbedding, PositionalEncoding, LayerNormalization, FeedForwardBlock, MultiHeadAttention, ResidualConnection, ProjectionLayer)
- `encoder.py` Bagian encoder dari model
- `decoder.py` Bagian decoder dari model
- `main.py` Full version dari model

## Dataset

Model ini akan dilatih menggunakan dataset terjemahan dari **Hugging Face Datasets**:
- Dataset: English-Indonesian translation pairs
- Format: Paired sentences dengan struktur source-target
- Preprocessing: Tokenisasi dan padding untuk kedua bahasa

## Konfigurasi Model

```python
# Default hyperparameters
d_model = 512        # Dimensi embedding
N = 6               # Jumlah layer encoder/decoder
h = 8               # Jumlah attention heads
dropout = 0.1       # Dropout rate
d_ff = 2048         # Dimensi feed forward
src_vocab_size = 10000  # Vocabulary size source
tgt_vocab_size = 10000  # Vocabulary size target
max_seq_len = 100   # Maximum sequence length
```

## Evaluasi Model

Metrik evaluasi yang akan digunakan:
- **BLEU Score**: Mengukur kualitas terjemahan dengan membandingkan n-gram
- **Perplexity**: Mengukur seberapa baik model memprediksi data test
- **Loss**: Cross-entropy loss selama training dan validation

## Training Pipeline

1. **Data Preprocessing**: Tokenisasi dan encoding
2. **Model Initialization**: Xavier uniform initialization
3. **Training Loop**: 
   - Forward pass dengan teacher forcing
   - Backpropagation dan optimization
   - Validation setiap epoch
4. **Inference**: Greedy decoding atau beam search

## Source
- Attention is All You Need Paper
- [Transformers from scratch by Umar Jamil](https://www.youtube.com/watch?v=ISNdQcPhsts&t=108s)
- [Pytorch Transformers by Aladdin Persson](https://www.youtube.com/watch?v=U0s0f995w14&t=1905s)
