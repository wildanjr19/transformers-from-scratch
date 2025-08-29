import torch
import torch.nn as nn
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel

# dapatkan semua data (get_all_sentences)
# latih tokenizer (get_or_build_tokenizer)

def get_all_sentences(ds, lang, two_col_map = None, translation_key = "translation"):
    """
    Ambil semua data dari dataset
    Args:
        - lang : en atau id
        - two_col_map : mapping jika menggunakan schema dua kolom
    """