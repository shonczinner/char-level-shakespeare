import argparse
import torch
import os
from torch.utils.data import DataLoader, Subset
from utils.tokenizer import CharTokenizer
from utils.dataset import ShakespeareDataset
from models import get_model
from utils.config import Config
from utils.plot_metrics import plot_metrics
import pandas as pd
import time


class Trainer:
    def __init__(self, config: Config):
        self.config = config

        self.save_path =  os.path.join(config.save_path, config.model_type)
        os.makedirs(self.save_path, exist_ok=True)
        self.model_path = os.path.join(self.save_path, "model.pth")

        with open(config.data_path, 'r') as f:
            self.text = f.read()

        if os.path.exists(os.path.join(self.save_path,"tokenizer.json")):
            self.tokenizer = CharTokenizer.load(self.save_path)
            print("Tokenizer loaded from", self.save_path)
        else: 
            self.tokenizer = CharTokenizer(self.text)
            self.tokenizer.save(self.save_path)
            print("Tokenizer saved to", self.save_path)

        self.config.vocab_size = self.tokenizer.vocab_size

        full_dataset = ShakespeareDataset(
            self.text,
            config.max_seq_len,
            self.tokenizer,
            stride=config.max_seq_len // 2,
            device=config.device
        )

        total_len = len(full_dataset)
        train_end = int(config.train_pct * total_len)
        val_end = train_end + int(config.val_pct * total_len)

        self.train_set = Subset(full_dataset, range(0, train_end))
        self.val_set = Subset(full_dataset, range(train_end, val_end))
        self.test_set = Subset(full_dataset, range(val_end, total_len))

        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=val_end - train_end)
        self.test_loader = DataLoader(self.test_set, batch_size=total_len - val_end)

        self.model = get_model(self.config).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()


        
        self.train_losses, self.val_losses, self.train_accs, self.val_accs,self.compute = [], [], [], [],[]
        self.load_model()

        
        

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.model = self.model.to(self.config.device)
            
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



    def run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        
        total_loss, total_correct, total_tokens = 0, 0, 0
        with torch.set_grad_enabled(train):
            start = time.time()
            for x, y in loader:
                if train:
                    self.optimizer.zero_grad()
 

                if self.config.model_type == 'rnn':
                    logits, _ = self.model(x) 
                else:
                    logits = self.model(x)
                    
                logits = logits.view(-1, self.config.vocab_size)
                y = y.view(-1)
                loss = self.loss_fn(logits, y)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_tokens += y.numel()
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
        plot_metrics(self.compute, self.train_losses, self.val_losses, self.save_path, "Loss", "Compute in seconds")
        plot_metrics(self.compute, self.train_accs, self.val_accs, self.save_path, "Accuracy", "Compute in seconds")


    def evaluate(self):
        test_loss, test_acc,_ = self.run_epoch(self.test_loader,False)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence models on Shakespeare data.")
    parser.add_argument('--model_type', type=str, default='rnn', help='Model type: rnn, cnn or transformer')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--train_pct', type=float, default=0.8)
    parser.add_argument('--val_pct', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_path', type=str, default='data/tinyshakespeare.txt')
    parser.add_argument('--save_path', type=str, default='experiments/')
    parser.add_argument('--kernel_size', type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = Config(**vars(args))
    trainer = Trainer(config)
    trainer.train()
    trainer.evaluate()
