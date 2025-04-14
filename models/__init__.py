# models/__init__.py
from .rnn import RNNModel
from .transformer import TransformerModel
from .cnn import CNNModel

def get_model(config):
    if config.model_type == 'rnn':
        return RNNModel(config)
    elif config.model_type == 'transformer':
        return TransformerModel(config)
    elif config.model_type == 'cnn':
        return CNNModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
