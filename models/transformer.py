import torch
import torch.nn as nn
from dataclasses import dataclass

from x_transformers import Decoder


@dataclass
class TransformerConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    nhead: int = 2
    use_rpe: bool = True


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, config):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.use_rpe = config.use_rpe

        self.embed = nn.Embedding(
            vocab_size,
            config.hidden_dim
        )

        if not self.use_rpe:
            # Learned absolute positional embeddings
            self.pos_emb = nn.Embedding(
                max_seq_len,
                config.hidden_dim
            )

        self.transformer = Decoder(
            dim=config.hidden_dim,
            depth=config.num_layers,
            heads=config.nhead,

            # Relative positional bias
            rel_pos_bias=config.use_rpe
        )

        self.fc = nn.Linear(
            config.hidden_dim,
            vocab_size
        )

    def forward(self, x):
        # x: [batch, seq_len]
        x = x[:, -self.max_seq_len:]

        B, T = x.shape

        x = self.embed(x)

        if not self.use_rpe:
            pos = torch.arange(
                T,
                device=x.device
            )

            x = x + self.pos_emb(pos)[None, :, :]

        x = self.transformer(x)

        return self.fc(x)
    
if __name__ == "__main__":
    # Small sanity check
    vocab_size = 1000
    max_seq_len = 32

    config = TransformerConfig(
        hidden_dim=128,
        num_layers=2,
        nhead=2,
        use_rpe=True
    )

    model = TransformerModel(
        vocab_size,
        max_seq_len,
        config
    )

    batch_size = 4
    seq_len = 16

    # Random token IDs
    x = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len)
    )

    logits = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)

    # Expected:
    # Input shape:  [4, 16]
    # Output shape: [4, 16, 1000]

    assert logits.shape == (
        batch_size,
        seq_len,
        vocab_size
    )

    # Check gradients flow
    loss = logits.mean()
    loss.backward()

    params_with_grad = sum(
        p.grad is not None
        for p in model.parameters()
    )

    total_params = sum(
        1
        for p in model.parameters()
    )

    print(f"Parameters with gradients: {params_with_grad}/{total_params}")
    print("Test passed!")