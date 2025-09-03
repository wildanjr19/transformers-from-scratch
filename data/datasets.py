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
    """"""
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
