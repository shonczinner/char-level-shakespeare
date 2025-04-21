# models/__init__.py
from .rnn import RNNModel
from .transformer import TransformerModel
from .cnn import CNNModel
from .gru import GRUModel
from .lstm import LSTMModel


def get_model(config):
    if config.model_type == 'rnn':
        return RNNModel(config)
    elif config.model_type == 'transformer':
        return TransformerModel(config)
    elif config.model_type == 'cnn':
        return CNNModel(config)
    elif config.model_type == 'gru':
        return GRUModel(config)
    elif config.model_type == 'lstm':
        return LSTMModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
