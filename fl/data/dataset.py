"""
MNIST dataset with IID and non-IID (Dirichlet) partitioning.
Non-IID simulates real-world federated settings where each client
has data from only a subset of classes (e.g., different hospitals,
different users' devices).
"""
import logging
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_mnist(data_dir: str = "./data/raw") -> Tuple:
    """Download and return full MNIST train/test datasets."""
    train = datasets.MNIST(data_dir, train=True,  download=True, transform=TRANSFORM)
    test  = datasets.MNIST(data_dir, train=False, download=True, transform=TRANSFORM)
    return train, test


def partition_iid(dataset, num_clients: int) -> List[List[int]]:
    """
    IID partitioning — shuffle and split equally.
    Each client gets a random uniform sample of all classes.
    """
    indices = np.random.permutation(len(dataset))
    return [indices[i::num_clients].tolist() for i in range(num_clients)]


def partition_non_iid(dataset, num_clients: int, alpha: float = 0.5) -> List[List[int]]:
    """
    Non-IID partitioning using Dirichlet distribution.

    alpha controls heterogeneity:
      alpha=0.1  → extremely non-IID (each client ≈ 1-2 classes)
      alpha=0.5  → moderately non-IID
      alpha=100  → nearly IID

    This simulates real FL scenarios where each device (client) has
    data that reflects its own usage patterns.
    """
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        # Dirichlet distribution over clients for this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_indices[c])).astype(int)
        # Fix rounding
        proportions[-1] = len(class_indices[c]) - proportions[:-1].sum()

        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(class_indices[c][start:start+count].tolist())
            start += count

    return client_indices


def get_client_loaders(
    dataset,
    client_indices: List[int],
    batch_size: int = 32,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """Return train and validation DataLoaders for a single client."""
    indices = np.array(client_indices)
    np.random.shuffle(indices)
    split = max(1, int(len(indices) * (1 - val_split)))
    train_idx = indices[:split].tolist()
    val_idx   = indices[split:].tolist()

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_test_loader(test_dataset, batch_size: int = 256) -> DataLoader:
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_class_distribution(dataset, indices: List[int]) -> dict:
    """Return class distribution for a client's data — useful for visualization."""
    labels = np.array(dataset.targets)[indices]
    unique, counts = np.unique(labels, return_counts=True)
    return {int(c): int(n) for c, n in zip(unique, counts)}
