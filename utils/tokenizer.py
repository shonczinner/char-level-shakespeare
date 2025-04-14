import json
import os

class CharTokenizer:
    def __init__(self, text=None):
        if text:
            self.chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(self.chars)
        else:
            self.chars = []
            self.stoi = {}
            self.itos = {}
            self.vocab_size = 0

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        tokenizer_data = {
            "chars": self.chars
        }
        with open(os.path.join(folder_path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, folder_path):
        with open(os.path.join(folder_path, "tokenizer.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.chars = data["chars"]
        tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
        tokenizer.itos = {i: ch for ch, i in tokenizer.stoi.items()}
        tokenizer.vocab_size = len(tokenizer.chars)
        return tokenizer
