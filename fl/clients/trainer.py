"""
Local client training for federated learning.
Each client trains on its local data for E epochs,
then sends model updates (not raw data) to the server.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fl.clients.model import MNISTNet
from fl.privacy.dp import DPConfig, clip_gradients

logger = logging.getLogger(__name__)


def train_local(
    model: MNISTNet,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.01,
    dp_config: Optional[DPConfig] = None,
    device: str = "cpu",
) -> Tuple[List[np.ndarray], Dict]:
    """
    Train model on local client data for E epochs.

    Args:
        model: current global model
        train_loader: client's local DataLoader
        epochs: local epochs (E in FedAvg paper)
        lr: learning rate
        dp_config: differential privacy config (optional)
        device: 'cpu' or 'mps' or 'cuda'

    Returns:
        (updated_parameters, metrics_dict)
    """
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    grad_norms = []

    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Differential privacy: clip gradients before update
            if dp_config and dp_config.enabled:
                norm = clip_gradients(model, dp_config.max_grad_norm)
                grad_norms.append(norm)

            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    metrics = {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "samples": total_samples,
        "epochs": epochs,
        "avg_grad_norm": round(float(np.mean(grad_norms)), 4) if grad_norms else None,
    }

    params = [val.cpu().numpy() for val in model.state_dict().values()]
    return params, metrics


def evaluate_local(
    model: MNISTNet,
    val_loader: DataLoader,
    device: str = "cpu",
) -> Dict:
    """Evaluate model on local validation set."""
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * len(batch_y)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

    return {
        "val_loss": round(total_loss / total_samples, 4),
        "val_accuracy": round(total_correct / total_samples, 4),
        "val_samples": total_samples,
    }


def evaluate_global(
    model: MNISTNet,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict:
    """Evaluate global model on centralized test set."""
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * len(batch_y)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)
            for label, pred in zip(batch_y, preds):
                class_correct[label] += (label == pred).item()
                class_total[label] += 1

    per_class = {str(i): round(class_correct[i]/class_total[i], 4) if class_total[i] > 0 else 0
                 for i in range(10)}

    return {
        "test_loss": round(total_loss / total_samples, 4),
        "test_accuracy": round(total_correct / total_samples, 4),
        "test_samples": total_samples,
        "per_class_accuracy": per_class,
    }
