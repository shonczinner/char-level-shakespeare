from dataclasses import dataclass, asdict
import json
import os

@dataclass
class Config:
    # Data and Training
    batch_size: int = 64
    max_seq_len: int = 128
    learning_rate: float = 3e-4
    epochs: int = 20
    train_pct: float = 0.8
    val_pct: float = 0.1
    test_pct: float = 0.1


    # Model
    model_type: str = 'rnn'

    # Apply to all Models
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    vocab_size: int = 0 # taken from tokenizer

    # CNN
    kernel_size: int = 5

    # Transformer
    nhead: int = 2

    def save(self, path: str):
        with open(os.path.join(path,"config.json"), 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load(path: str) -> "Config":
        with open(os.path.join(path,"config.json"), 'r') as f:
            data = json.load(f)
        return Config(**data)
        
