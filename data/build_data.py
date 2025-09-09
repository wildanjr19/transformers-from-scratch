import torch
from torch.utils.data import DataLoader, random_split
from data.datasets import load_dataset
from tokenizer_utils import get_or_build_tokenizer
from .datasets import BilingualDataset

"""Data loader (train dan validation)"""

def build_loaders_and_tokenizers(config):
    # load raw dataset
    raw_ds = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split = "train"
    )

    # apply tokenizer
    tokenizer_src = get_or_build_tokenizer(config, raw_ds, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, raw_ds, config['lang_tgt'])

    # split dataset menjadi train (90%) dan validasi (10%)
    train_size = int(0.9 * len(raw_ds))
    val_size = len(raw_ds) - train_size

    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_size, val_size])

    # apply BilingualDataset
    train_ds = BilingualDataset()