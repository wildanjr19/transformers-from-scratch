"""
Script untuk proses validasi model selama pelatihan
Evaluasi terhadap beberapa metrik -> BLEU, WER, dan CER

    run_validation  : main function untuk proses validasi
    greedy_decode   : fungsi untuk dekripsi dengan metode greedy
"""

import torch
import torch.nn as nn
from datasets import causal_mask

from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_msk, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Fungsi untuk dekripsi dengan metode greedy (search)
    """
    # get id of sos and eos
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # dapatkan encoder_output yang bisa digunakan di setiap langkah decoding
    encoder_output = model.encode(source, source_msk)

    # inisialisasi decoder_input dengan token [SOS]
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # buat mask untuk target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)

        # dapatkan output dari model
        output = model.decode(encoder_output, source_msk, decoder_input, decoder_mask)

        # dapatkan next token dengan probabilitas (logit) tertinggi
        logit = model.proj_out(output[:, -1])  # ambil output terakhir
        _, next_token = torch.max(logit, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_token.item()).to(device)], dim=1
        )

        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)

def validate_model(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
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

            # dapatkan model output dengan greedy decode
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]

            # konversi token hasil prediksi ke text
            model_output_tokens = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected_texts.append(target_text)
            predicted_texts.append(model_output_tokens)

            # ambil satu contoh
            # use print_msg bcs use tqdm in training
            print_msg('-' * console_width)
            print_msg(f"SUMBER: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDIKSI: {model_output_tokens}")

            if count >= num_examples:
                break