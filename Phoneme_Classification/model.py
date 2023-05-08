import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Utility_Functions import *
from torch.utils.data import DataLoader
import gc


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, lambd=1):
        super(BasicBlock, self).__init__()

        # TODO: apply batch normalization and dropout for strong baseline.
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
            # nn.Dropout(lambd)
        )

    def forward(self, x):
        x = self.block(x)
        return x

# lambd has an error
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim=256, lambd=1) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            *[BasicBlock(output_dim, output_dim, lambd) for _ in range(3)]
        )
    
    def forward(self, x):
        residual = x
        out=self.block(x)+residual
        return out


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, lambd=0.5):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim, lambd) for _ in range(hidden_layers)],
            # *[ResidualBlock(hidden_dim,hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

