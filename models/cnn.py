import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class CNN(nn.Module):
    def __init__(self, input_dim,hidden_dim,kernel_size,num_layers):
        super().__init__()
        layers = []
        in_channels = input_dim
        for _ in range(num_layers):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size-1 
            )
            layers.append(conv)
            layers.append(Chomp1d(kernel_size-1))
            layers.append(nn.ReLU())
            in_channels = hidden_dim

        self.conv_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        x = x.transpose(1, 2)  # [batch, embed_dim, seq_len] for Conv1d
        x = self.conv_layers(x)  # [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        return x

class CNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)

        self.cnn = CNN(config.embed_dim,config.hidden_dim,config.kernel_size,config.num_layers)

        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embed(x)  # [batch, seq_len, embed_dim]
        x = self.cnn(x) # [batch, seq_len, hidden_dim]
        logits = self.fc(x)  # [batch, seq_len, vocab_size]
        return logits