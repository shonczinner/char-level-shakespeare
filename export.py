import torch
import os

from models import get_model
from utils.tokenizer import CharTokenizer
from utils.config import Config
from utils.constants import (PROCESSED_DATA_PATH,
                             SAVE_PATH,
                             TOKENIZER_PATH)

# Load config and tokenizer
model_folder = os.path.join(SAVE_PATH,'rnn')
config = Config.load(model_folder)    



# Load model
model = get_model(config)

model_path = os.path.join(model_folder,"model.pth")
model.load_state_dict(torch.load(model_path, map_location='cpu'))
print("Model loaded from:",model_path)
# Create a dummy input matching your model's input shape
# Let's assume character-level input with shape [1, seq_len]
dummy_input = torch.zeros(1, 1).long()  # adjust if your model takes different shape
dummy_hidden = torch.zeros(config.num_layers, 1, config.hidden_dim)

# Export to ONNX with hidden state
torch.onnx.export(
    model,
    (dummy_input, dummy_hidden),  # Pass both input and hidden state
    "model.onnx",
    input_names=["input", "hidden"],  # Input names for both input and hidden state
    output_names=["output", "hidden"],  # Output names for both logits and hidden state
    dynamic_axes={
        "input": {1: "seq_len"},  # Make seq_len dynamic
    },
    opset_version=11
)
#################


import json
tokenizer = CharTokenizer.load(TOKENIZER_PATH)

chars = tokenizer.chars
vocab = {ch: i for i, ch in enumerate(chars)}

# Save vocab.json
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

