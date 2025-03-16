import torch
import torch.nn as nn
from config import Config

class REINFORCE(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2) -> None:
        super(REINFORCE, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_dim, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, act_dim*2))  # act_dim * (1 for mean + 1 for std)
        self.model = nn.Sequential(*layers)
        self.to(Config.device)  # Move the model to the specified device

    def forward(self, x):
        return self.model(x)
    