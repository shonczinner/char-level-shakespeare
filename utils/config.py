from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Config:
    # Training
    batch_size: int = 64
    max_seq_len: int = 128
    learning_rate: float = 3e-4
    epochs: int = 20

    # Dataset
    train_pct: float = 0.8
    val_pct: float = 0.1
    test_pct: float = 0.1

    # Model
    model_type: str = "rnn"
    vocab_size: int = 0


    def save(self, path: str | Path):
        path = Path(path)

        if path.suffix != ".json":
            path = path / "config.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(
                asdict(self),
                f,
                indent=4,
            )

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with path.open("r") as f:
            data = json.load(f)

        return cls(**data)