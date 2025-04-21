import torch
from torch.utils.data import Dataset, DataLoader, Subset

class ShakespeareDataset(Dataset):
    def __init__(self, data, seq_len,stride):
        self.seq_len = seq_len
        self.stride = stride

        self.data = data

        # Precompute start indices for sequences
        self.start_idxs = list(range(0, len(self.data) - seq_len - 1, self.stride))

    def __len__(self):
        return len(self.start_idxs)

    def __getitem__(self, idx):
        i = self.start_idxs[idx]
        x = self.data[i:i + self.seq_len]
        y = self.data[i + 1:i + self.seq_len + 1]
        return x, y


def get_loaders(full_dataset, batch_size, train_pct, val_pct):
    total_len = len(full_dataset)
    train_end = int(train_pct * total_len)
    val_end = train_end + int(val_pct * total_len)

    train_set = Subset(full_dataset, range(0, train_end))
    val_set = Subset(full_dataset, range(train_end, val_end))
    test_set = Subset(full_dataset, range(val_end, total_len))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_end - train_end)
    test_loader = DataLoader(test_set, batch_size=total_len - val_end)

    return train_loader, val_loader, test_loader