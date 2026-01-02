"""
Script utama untuk proses pelatihan model

    get_model   : fungsi untuk membangun model dari script transformers_model.py
    train_model : fungsi utama untuk proses pelatihan model
"""

import torch
import torch.nn as nn

import torchtext.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

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

    initial_epoch = 0
    global_step = 0
    
    # preload weights/training jika ada atau checkpoint
    if config['preload']:
        model_filename = get_weights_file(config, config['preload'])
        print(f"Preload model from {model_filename}")
        # load state
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        # load optimizer state
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # loss function
    # PAD di ignore supaya tidak dihitung dalam loss
    # label smoothing untuk regularisasi
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)

    # main train loop
    for epoch in range(initial_epoch, config['num_epochs']):
        # free memory first
        torch.cuda.empty_cache()
        model.train()
        # buat tqdm untuk pantau proses di tiap batch
        train_batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        # loop tiap batch
        for batch in train_batch_iterator:
            # ambil/siapkan data input dari batch
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (b, 1, seq_len, seq_len)

            # masukkan semua data input ke model
            encoder_output = model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (b, seq_len, d_model)
            projection_output = model.projection(decoder_output) # (b, seq_len, vocab_size)

            # dapatkan label/target dari batch
            label = batch['label'].to(device) # (b, seq_len)

            # hitung loss
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            # logging
            writer.add_scalar('Train Loss', loss.item(), global_step)
            writer.flush()

            # backprop
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        # simpan model tiap epoch
        model_filename = get_weights_file(config, f"{epoch:02d}")
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)