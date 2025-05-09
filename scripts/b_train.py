import argparse
import torch
import os
from utils.tokenizer import CharTokenizer
from utils.dataset import ShakespeareDataset, get_loaders
from models import get_model
from utils.config import Config
from utils.plot_metrics import plot_metrics
import pandas as pd
import time
from constants import (PROCESSED_DATA_PATH,
                             SAVE_PATH,
                             TOKENIZER_PATH)



class Trainer:
    def __init__(self, data, config: Config, device):
        self.config = config
        self.device = device

        self.train_loader, self.val_loader, self.test_loader = get_loaders(
            data,
            config.max_seq_len,
            config.max_seq_len//2,
            config.batch_size,
            config.train_pct,
            config.val_pct
        )

        self.save_path =  os.path.join(SAVE_PATH, config.model_type)
        os.makedirs(self.save_path, exist_ok=True)
        self.model_path = os.path.join(self.save_path, "model.pth")

        self.model = get_model(self.config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_losses, self.val_losses, self.train_accs, self.val_accs,self.compute = [], [], [], [],[]
        self.load_model()


    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.model = self.model.to(self.device)
            
            metrics = pd.read_csv(os.path.join(self.save_path, "metrics.csv"),header=0)
            self.train_losses = metrics["train_losses"].tolist()
            self.val_losses = metrics["val_losses"].tolist()
            self.train_accs = metrics["train_accs"].tolist()
            self.val_accs = metrics["val_accs"].tolist()
            self.compute = metrics["compute"].tolist()

            print("Model loaded from",self.model_path)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        metrics = pd.DataFrame({
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "compute":self.compute
            })
        metrics.to_csv(os.path.join(self.save_path, "metrics.csv"), index=False)

        self.config.save(self.save_path)

        print("Model saved at",self.model_path)



    def run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        
        total_loss, total_correct, total_tokens = 0, 0, 0
        with torch.set_grad_enabled(train):
            start = time.time()
            for x, y in loader:
                if train:
                    self.optimizer.zero_grad()
 

                if self.config.model_type in ['rnn','gru','lstm','mingru']:
                    logits, _ = self.model(x) 
                else:
                    logits = self.model(x)
                    
                logits = logits.view(-1, self.config.vocab_size)
                y = y.view(-1)
                loss = self.loss_fn(logits, y)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.detach().item()
                preds = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().detach().item()
                total_tokens += y.detach().numel()
        end = time.time()
        return total_loss / len(loader), total_correct / total_tokens, end-start

    def train(self):
        if len(self.compute)>0:
            compute = self.compute[-1]
        else:
            compute = 0
        for epoch in range(self.config.epochs):
            train_loss, train_acc,compute_i = self.run_epoch(self.train_loader,True)
            val_loss, val_acc,_ = self.run_epoch(self.val_loader,False)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Time: {compute_i:.4f} seconds")
            print(f" Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            compute = compute+compute_i
            self.compute.append(compute)

        self.save_model()
        plot_metrics(None, self.train_losses, self.val_losses, self.save_path, "Loss")
        plot_metrics(None, self.train_accs, self.val_accs, self.save_path, "Accuracy")
        plot_metrics(self.compute, self.train_losses, self.val_losses, self.save_path, "Loss", "Train compute in seconds")
        plot_metrics(self.compute, self.train_accs, self.val_accs, self.save_path, "Accuracy", "Train compute in seconds")


    def evaluate(self):
        test_loss, test_acc,_ = self.run_epoch(self.test_loader,False)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence models on Shakespeare data.")

    parser.add_argument('--model_type', type=str, default='rnn')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=3e-4)


    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--nhead', type=int, default=2)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = Config(**vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    config.vocab_size = tokenizer.vocab_size

    with open(PROCESSED_DATA_PATH, 'r') as f:
        tokens_s = f.read()

    tokens = [int(x) for x in tokens_s.split(',')]
    data = torch.tensor(tokens, dtype=torch.long, device=device)

    trainer = Trainer(data,config, device)
    trainer.train()
    trainer.evaluate()
