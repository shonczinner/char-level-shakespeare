from .rnn import RNNModel, RNNConfig
from .gru import GRUModel, GRUConfig
from .lstm import LSTMModel, LSTMConfig
from .cnn import CNNModel, CNNConfig
from .transformer import TransformerModel, TransformerConfig


def get_model(config):
    if config.model_type == "rnn":
        return RNNModel(
            vocab_size=config.vocab_size,
            config=RNNConfig(),
        )

    elif config.model_type == "gru":
        return GRUModel(
            vocab_size=config.vocab_size,
            config=GRUConfig(),
        )

    elif config.model_type == "lstm":
        return LSTMModel(
            vocab_size=config.vocab_size,
            config=LSTMConfig(),
        )

    elif config.model_type == "cnn":
        return CNNModel(
            vocab_size=config.vocab_size,
            config=CNNConfig(),
        )

    elif config.model_type == "transformer":
        return TransformerModel(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            config=TransformerConfig(),
        )

    else:
        raise ValueError(f"Unknown model type: {config.model_type}")