import torch
import os
import json
from models import get_model
from utils.tokenizer import CharTokenizer
from utils.config import Config
from constants import (PROCESSED_DATA_PATH,
                             SAVE_PATH,
                             TOKENIZER_PATH)

# Load config and tokenizer
model_folder = os.path.join(SAVE_PATH,'rnn')
config = Config.load(model_folder)    



# Load model
model = get_model(config)
model.eval()

model_path = os.path.join(model_folder,"model.pth")
model.load_state_dict(torch.load(model_path, map_location='cpu'))
print("Model loaded from:",model_path)
# Create a dummy input matching your model's input shape
# Let's assume character-level input with shape [1, seq_len]
dummy_input = torch.zeros(1, 1).long()  # adjust if your model takes different shape
dummy_hidden = model(dummy_input)[1].detach()  # Get the initial hidden state from the model

# Export to ONNX with hidden state
torch.onnx.export(
    model,
    (dummy_input, dummy_hidden),
    "model.onnx",
    input_names=["input", "hidden"],
    output_names=["output", "hidden"],
    dynamic_shapes=(
        {1: "seq_len"},
        None,
    ),
    opset_version=18,
)
#################
tokenizer = CharTokenizer.load(TOKENIZER_PATH)

chars = tokenizer.chars
vocab = {ch: i for i, ch in enumerate(chars)}

# Save vocab.json
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

