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