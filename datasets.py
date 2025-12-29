import torch
from torch.utils.data import Dataset

# causal masking
# dataset class

def causal_mask(size: int):
    """Buat matriks untuk masking"""
    # [bh, 1, size, size]
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    """Kelas dataset"""
    def __init__(self, ds, tokenizer_src,  tokenizer_tgt, src_lang, tgt_lang, seq_len) :
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # token khusus
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # konversi text ke token dan akses ID token
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # tambahkan token khusus (sos, eos, dan pad)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 untuk [SOS] dan [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 

        # pastikan panjang token padding tidak negatif, jika negatif kembalikan eror terlalu panjang
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Input terlalu panjang")
        
        # buat encoder input dengan menambahkan sos, eos, dan pad
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # buat decoder input dengan menambahkan sos dan pad
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] *dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # buat label/target dengan menambahkan eos dan pad
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # pastikan kembali semua ukuran input sesuai dengan seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # kembalikan sebagai dictionary
        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label" : label, # seq_len
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }
        

