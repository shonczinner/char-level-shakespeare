import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            dropout=0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        x = x[:, -self.max_seq_len:] # Truncate to max context
        B, T = x.size()

        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        x = self.embed(x) + self.pos_emb(pos) # [B, T, hidden_dim]

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask) # Causal mask

        return self.fc(x) # [B, T, vocab_size]
