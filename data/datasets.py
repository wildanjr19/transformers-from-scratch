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
    
    # karena dataset kita terdiri dari dua kolom, maka buat class dataset agar paham dua kolom
    def __getitem__(self, index):
        row = self.ds[index]
        # semisal sudah ada pada satu kolom translation
        if "translation" in row:
            src_text = row["translation"][self.src_lang]
            tgt_text = row["translation"][self.tgt_lang]
        # jika ada dua kolom bahasa
        else:
            # pastikan self.src_lang='en', self.tgt_lang='id'
            col_map = {"en" : "english", "id" : "indonesian"}
            src_text = row[col_map[self.src_lang]]
            tgt_text = row[col_map[self.tgt_lang]]

        # ubah dari teks menjadi token
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        """Kurangi kalimat untuk menambahkan token khusus<s> dan </s> untuk input, dan hanya </s> untuk target"""
        # encoder (<s> dan </s>)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # decoder (</s>)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # pastikan padding token tidak negatif, karena bisa saja teksnya terlalu panjang
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Kalimat terlalu panjang")
        
        """Menambahkan token khusus"""
        # tambahkan token khusus encoder
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # tambahkan token khusus decoder
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # target hanya </s> dan padding
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # pastikan ukuran sudah sesuai dengan seq_len -> [seq_len]
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # kembalikan dalam bentuk dictinary
        """encoder_mask dan decoder_mask masking boolean"""
        return {
            "encoder_input" : encoder_input,                                                    # [seq_len]
            "decoder_input" : decoder_input,                                                    # [seq_len]                             
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # [1, 1, seq_len]
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # [1, seq_len] & [1, seq_len, seq_len]
            "label" : label, # [seq_len]
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }

