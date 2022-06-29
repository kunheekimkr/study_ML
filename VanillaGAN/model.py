from audioop import lin2adpcm
from statistics import LinearRegression
from torch import dropout
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().init()
        self.model = nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256,1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.model(x)
        return x