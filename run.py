import argparse
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence models on Shakespeare data.")

    parser.add_argument(
    "--model_types",
    nargs="+",
    default=["rnn", "cnn", "transformer","gru","lstm"],
    help="List of model types to compare (default: rnn cnn transformer)"
    )

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=3e-4)

    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--nhead', type=int, default=2)
    
    return parser.parse_args()


if __name__=="__main__":
    import torch

    from scripts.a_tokenize_data import tokenize_data
    from scripts.b_train import Trainer
    from scripts.c_compare_experiments import compare_experiments
    from utils.tokenizer import CharTokenizer
    from utils.constants import (PROCESSED_DATA_PATH,
                             SAVE_PATH)
    from utils.dataset import ShakespeareDataset
    from utils.config import Config

    args = parse_args()
    args_dict = vars(args)
    model_types = args_dict.pop("model_types", None)  # Safely removes it if it exists
    config = Config(**args_dict)

    tokenizer, tokens = tokenize_data()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device=  'cpu'

    data = torch.tensor(tokens, dtype=torch.long, device=device)
    dataset = ShakespeareDataset(data,config.max_seq_len,config.max_seq_len//2)

    # train models

    for model_type in model_types:
        print(f"training {model_type}")
        config.model_type = model_type
        config.vocab_size = tokenizer.vocab_size
        trainer = Trainer(dataset,config, device)
        trainer.train()
        trainer.evaluate()
        
        # Clear CUDA memory
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    # plot all metrics
    compare_experiments(model_types)


    

    
