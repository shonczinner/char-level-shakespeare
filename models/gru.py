import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)

        self.gru = nn.GRU(config.embed_dim, config.hidden_dim, config.num_layers, batch_first=True)

        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        # h: None or [num_layers, batch, hidden_size]
        x = self.embed(x) # [batch, seq_len, embed_dim]
        out, hidden = self.gru(x, hidden) # [batch, seq_len, hidden],[num_layers, batch, hidden_size]
        logits = self.fc(out) # [batch, seq_len, vocab_size]
        return logits, hidden # [batch, seq_len, vocab_size], [num_layers, batch, hidden_size]
