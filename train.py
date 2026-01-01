import torch
import torch.nn as nn

import torchtext.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

# our scripts
from transformers_model import build_transformers
from datasets import BilingualDataset, causal_mask
from config import get_config, get_weights_file
from build_data import get_dataset

# HF utils
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# build model function
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformers(
        vocab_src_len,
        vocab_tgt_len, 
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    )
    return model

# training function
def train_model(config):
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # buat folder untuk model
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # get dataset
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_dataset(config)

    # build model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # tensorboard writer -> untuk logging proses
    writer = SummaryWriter(config['experiment_name'])

    # optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)