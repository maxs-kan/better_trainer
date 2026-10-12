import torch.nn as nn


class UpMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.ffn(x)


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            UpMLP(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            UpMLP(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
            )
    def forward(self, x):
        return self.net(x)