import torch
import torch.nn as nn

import torchtext.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# our scripts
from transformers_model import build_transformers
from datasets import BilingualDataset, causal_mask
from config import get_config, get_weights_file

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