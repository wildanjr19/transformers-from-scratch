"""
Script untuk membuat/mempersiapkan dataset
"""

import torch
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tokenizer_utils import get_or_build_tokenizer
from datasets import BilingualDataset

def get_dataset(config):
    """
    Get dataset untuk training, bangun tokenizer, split dataset (train dan val), buat DataLoader
    """
    # dataset hanya ada train, jadi kita split manual
    raw_ds = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split = 'train'
    )

    # buat tokenizer
    tokenizer_src = get_or_build_tokenizer(config, raw_ds, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, raw_ds, config['lang_tgt'])

    # split dataset menjadi train (90%) dan validasi (10%)
    train_size = int(0.9 * len(raw_ds))
    val_size = len(raw_ds) - train_size
    # random split
    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_size, val_size])

    # apply BilingualDataset
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # temukan panjang maksimum setiap kalimat di source dan target
    max_len_src = 0
    max_len_tgt = 0

    for item in raw_ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print max length
    print(f"Maksimum panjang kalimat sumber: {max_len_src}")
    print(f"Maksimum panjang kalimat target: {max_len_tgt}")

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt