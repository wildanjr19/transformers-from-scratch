"""
Script untuk proses validasi model selama pelatihan
Evaluasi terhadap beberapa metrik -> BLEU, WER, dan CER

    run_validation  : main function untuk proses validasi
    greedy_decode   : fungsi untuk dekripsi dengan metode greedy
"""

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_msk, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Fungsi untuk dekripsi dengan metode greedy
    """

def validate_model(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    """
    Fungsi untuk validasi pelatihan
    Args:
        model              : model yang akan divalidasi
        validation_dataset : dataset validasi
        tokenizer_src      : tokenizer untuk bahasa sumber
        tokenizer_tgt      : tokenizer untuk bahasa target
        max_len            : panjang maksimum urutan output
        device             : device 
        print_msg          : fungsi print
        global_state       : state global pelatihan
        writer             : SummaryWriter untuk TensorBoard
        num_examples       : jumlah contoh untuk ditampilkan
    """
    # set model ke eval
    model.eval()

    count = 0

    # inisialisasi sampel
    source_texts = []       # teks sumber
    expected_texts = []     # teks target
    predicted_texts = []    # teks hasil prediksi

    console_width = 80

    # loop val_ds
    with torch.no_grad():
        for batch in validation_dataset:
            count += 1

            # ambil data dari batch (encoder), pindah ke device
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # make sure ukuran batch = 1
            assert encoder_input.size(0) == 1, "Ukuran batch harus 1 untuk validasi"