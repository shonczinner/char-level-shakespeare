from dataclasses import dataclass, asdict
import json
import os

@dataclass
class Config:
    model_type: str = 'rnn'
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    vocab_size: int = 0
    max_seq_len: int = 128

    train_pct: float = 0.8
    val_pct: float = 0.1
    test_pct: float = 0.1

    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 20
    device: str = 'cuda'

    data_path: str = 'data/tinyshakespeare.txt'
    save_path: str = 'experiments/'

    kernel_size: int = 3

    def save(self, path: str):
        with open(os.path.join(path,"config.json"), 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load(path: str) -> "Config":
        with open(os.path.join(path,"config.json"), 'r') as f:
            data = json.load(f)
        return Config(**data)
        
