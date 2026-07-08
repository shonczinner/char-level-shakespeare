import argparse
import gc

import torch
from dataclasses import replace

from scripts.a_tokenize_data import tokenize_data
from scripts.b_train import Trainer
from scripts.c_compare_experiments import compare_experiments
from utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sequence models on Shakespeare data."
    )

    parser.add_argument(
        "--model_types",
        nargs="+",
        default=[
            "rnn",
            "cnn",
            "transformer",
            "gru",
            "lstm",
        ],
    )

    for field in Config.__dataclass_fields__.values():
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=None,
        )

    return parser.parse_args()


def load_config():
    config = Config()
    args = parse_args()

    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    model_types = args.model_types

    return config, model_types


def train_models(data, tokenizer, model_types, config, device):
    config = replace(config, vocab_size=tokenizer.vocab_size)

    for model_type in model_types:
        print(f"Training {model_type}")

        model_config = replace(
            config,
            model_type=model_type,
        )

        trainer = Trainer(
            data,
            model_config,
            device,
        )

        trainer.train()
        trainer.evaluate()

        del trainer
        torch.cuda.empty_cache()
        gc.collect()


def main():
    config, model_types = load_config()

    tokenizer, tokens = tokenize_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device,
    )

    train_models(
        data,
        tokenizer,
        model_types,
        config,
        device,
    )

    compare_experiments(model_types)


if __name__ == "__main__":
    main()