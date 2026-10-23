"""Compact policy-value residual CNN."""
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(y)))

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=9, channels=64, blocks=3):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(2 * board_size**2, board_size**2))
        self.value = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.ReLU(), nn.Flatten(), nn.Linear(board_size**2, channels), nn.ReLU(), nn.Linear(channels, 1), nn.Tanh())

    def forward(self, x):
        features = self.trunk(self.stem(x))
        return self.policy(features), self.value(features).squeeze(1)
