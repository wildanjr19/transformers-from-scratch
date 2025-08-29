import torch
import torch.nn as nn
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
from pathlib import Path

# dapatkan semua data (get_all_sentences)
# latih tokenizer (get_or_build_tokenizer)

# schema dataset ada dua kolom ['english', 'indonesian']

def get_all_sentences(ds, lang, two_col_map = None, translation_key = "translation"):
    """
    Ambil semua data dari dataset
    Args:
        - lang : en atau id
        - two_col_map : mapping jika menggunakan schema dua kolom
    """
    # default mapping
    if two_col_map is None:
        two_col_map = { "en": "english", "id": "indonesian"}

    col_name = two_col_map.get(lang, lang)

    for item in ds:
        text = None
        if translation_key in item and isinstance(item[translation_key], dict) and lang in item[translation_key]:
            text = item[translation_key][lang]
        elif col_name in item:
            text = item[col_name]
        elif lang in item:
            text = item[lang]

        if text:
            t = text.strip()
            if t:
                yield t

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang, two_col_map={"en": "english", "id": "indonesian"}), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer