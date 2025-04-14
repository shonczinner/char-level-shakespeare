import torch
import torch.nn as nn
from models.cnn import CNNModel  # Adjust path to where CNNModel is
from utils.config import Config  # Adjust as needed

def get_test_config(vocab_size=100, seq_len=32):
    return Config(
        model_type='cnn',
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        embed_dim=64,
        hidden_dim=128,
        num_layers=3,
        kernel_size=3
    )

def test_cnn_model_output_shape():
    config = get_test_config()
    model = CNNModel(config)

    batch_size = 4
    seq_len = config.max_seq_len
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(x)
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected output shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"

def test_variable_sequence_lengths():
    config = get_test_config()
    model = CNNModel(config)

    for seq_len in [8, 16, 64]:
        x = torch.randint(0, config.vocab_size, (2, seq_len))
        logits = model(x)
        assert logits.shape == (2, seq_len, config.vocab_size), f"Failed at seq_len={seq_len}"

def test_multiple_batch_sizes():
    config = get_test_config()
    model = CNNModel(config)

    for batch_size in [1, 8, 32]:
        x = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
        logits = model(x)
        assert logits.shape == (batch_size, config.max_seq_len, config.vocab_size)

def test_forward_and_backward():
    config = get_test_config()
    model = CNNModel(config)

    x = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    target = torch.randint(0, config.vocab_size, (2, config.max_seq_len))

    logits = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, config.vocab_size), target.view(-1))
    loss.backward()

    # Check that some gradients are non-zero
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "No gradients computed!"

if __name__ == "__main__":
    test_cnn_model_output_shape()
    test_variable_sequence_lengths()
    test_multiple_batch_sizes()
    test_forward_and_backward()
    print("All tests passed!")
