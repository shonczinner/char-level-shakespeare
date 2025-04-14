import torch
import argparse
from utils.tokenizer import CharTokenizer
from utils.config import Config
from models import get_model



@torch.no_grad()
def sample(model, tokenizer, config, start_text="ROMEO:", max_length=300, temperature=1.0):
    model.eval()
    model.to(config.device)

    input_ids = torch.tensor(tokenizer.encode(start_text), dtype=torch.long).unsqueeze(0).to(config.device)
    generated = input_ids

    hidden = None
    for _ in range(max_length):
        input_chunk = generated[:, -config.max_seq_len:]  # keep input size bounded
        if config.model_type == 'rnn':
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
    
    # Load config and tokenizer
    config = Config.load(os.path.join("experiments",args.model_type))
    save_path = os.path.join(config.save_path,args.model_type)

    with open(config.data_path, 'r') as f:
        text = f.read()
    
    if os.path.exists(os.path.join(save_path,"tokenizer.json")):
        tokenizer= CharTokenizer.load(save_path)
        print("Tokenizer loaded from", save_path)
    else: 
        print("Tokenizer not found")
        return

    # Load model
    model = get_model(config)

    model_path = os.path.join(save_path,"model.pth")
    model.load_state_dict(torch.load(model_path, map_location=config.device))

    # Sample
    output = sample(model, tokenizer, config, args.start, args.length, args.temp)
    print(output)

if __name__ == "__main__":
    main()
