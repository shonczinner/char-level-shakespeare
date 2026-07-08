import torch.nn as nn
from dataclasses import dataclass

@dataclass
class RNNConfig:
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2

class RNNModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, config.embed_dim)

        self.rnn = nn.RNN(config.embed_dim, config.hidden_dim, config.num_layers, batch_first=True)

        self.fc = nn.Linear(config.hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        # h: None or [num_layers, batch, hidden_size]
        x = self.embed(x) # [batch, seq_len, embed_dim]
        out, hidden = self.rnn(x, hidden) # [batch, seq_len, hidden],[num_layers, batch, hidden_size]
        logits = self.fc(out) # [batch, seq_len, vocab_size]
        return logits, hidden # [batch, seq_len, vocab_size], [num_layers, batch, hidden_size]
