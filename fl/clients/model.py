"""
CNN model for MNIST image classification.
Lightweight enough to train on CPU, representative of
on-device models used in real federated learning deployments.
"""
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    2-layer CNN for MNIST.
    Architecture mirrors what Apple uses for on-device models:
    small, fast, privacy-friendly.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28 → 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 14x14 → 14x14
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 28→14
        x = self.pool(F.relu(self.conv2(x)))   # 14→7
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

    def get_parameters(self) -> List[torch.Tensor]:
        return [val.cpu() for val in self.state_dict().values()]

    def set_parameters(self, parameters: List[torch.Tensor]) -> None:
        state = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), parameters)}
        )
        self.load_state_dict(state, strict=True)


def get_model() -> MNISTNet:
    return MNISTNet()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
