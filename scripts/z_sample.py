import torch
import argparse
from utils.tokenizer import CharTokenizer
from utils.config import Config
from models import get_model
from utils.constants import (TOKENIZER_PATH,
                             SAVE_PATH)


@torch.no_grad()
def sample(model, tokenizer, config, device, start_text="ROMEO:", max_length=300, temperature=1.0):
    model.eval()

    input_ids = torch.tensor(tokenizer.encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    generated = input_ids

    hidden = None
    for _ in range(max_length):
        input_chunk = generated[:, -config.max_seq_len:]  # keep input size bounded
        if config.model_type == 'rnn':
            #technicially could do: logits[:,-1:,:], hidden = model(input_chunk, hidden)
            logits, hidden = model(input_chunk, hidden)
        else:
            logits = model(input_chunk)

        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0].tolist())

def main():
    import os
    parser = argparse.ArgumentParser(description="Run sequence models on new text.")
    parser.add_argument('--model_type', type=str, default='rnn', help='Model type: rnn, cnn or transformer')
    parser.add_argument("--start", type=str, default="ROMEO:")
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config and tokenizer
    model_folder = os.path.join(SAVE_PATH,args.model_type)
    config = Config.load(model_folder)    
    
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
 
    # Load model
    model = get_model(config)

    model_path = os.path.join(model_folder,"model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Sample
    output = sample(model, tokenizer, config, device, args.start, args.length, args.temp)
    print(output)

if __name__ == "__main__":
    main()
