'''
Implementasi Self Attention Mechanism
Kelas Self Attention disini hanya sebagai contoh implementasi dari mekanisme self-attention
'''
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super(SelfAttention, self).__init__()

        # inisialisasi bobot Q, K , V
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)


    def forward(self, x):
        # Hitung nilai Q, K, V (dilewatkan bias)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # attention scores (nominator)
        attention_scores = queries @ keys.T

        # softmax dan normalisasi
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)

        # kalikan dengan V
        context_vector = attention_weights @ values

        return context_vector
    

# Example
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1) 
   [0.55, 0.87, 0.66], # journey  (x^2) 
   [0.57, 0.85, 0.64], # starts   (x^3) 
   [0.22, 0.58, 0.33], # with     (x^4) 
   [0.77, 0.25, 0.10], # one      (x^5) 
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(789)
selfatt = SelfAttention(d_in=3, d_out=3)
output = selfatt(inputs)
print(output)