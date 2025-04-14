import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_len, tokenizer,stride, device='cuda'):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride

        # Encode the whole dataset and move to device
        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long, device=device)

        # Precompute start indices for sequences
        self.start_idxs = list(range(0, len(self.data) - seq_len - 1, self.stride))

    def __len__(self):
        return len(self.start_idxs)

    def __getitem__(self, idx):
        i = self.start_idxs[idx]
        x = self.data[i:i + self.seq_len]
        y = self.data[i + 1:i + self.seq_len + 1]
        return x, y
