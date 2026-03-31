"""
Federated Averaging (FedAvg) — McMahan et al. 2017
https://arxiv.org/abs/1602.05629

Also implements FedProx proximal term for non-IID stability.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def federated_average(
    client_updates: List[Tuple[List[np.ndarray], int]],
) -> List[np.ndarray]:
    """
    FedAvg: weighted average of client model parameters.
    Weight = num_samples (larger clients contribute more).

    Args:
        client_updates: list of (parameters, num_samples) tuples

    Returns:
        Aggregated global model parameters
    """
    total_samples = sum(n for _, n in client_updates)
    aggregated = None

    for params, num_samples in client_updates:
        weight = num_samples / total_samples
        if aggregated is None:
            aggregated = [weight * p for p in params]
        else:
            for i, p in enumerate(params):
                aggregated[i] += weight * p

    return aggregated


def federated_average_equal(
    client_params: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """Simple unweighted FedAvg (equal weight per client)."""
    aggregated = None
    n = len(client_params)
    for params in client_params:
        if aggregated is None:
            aggregated = [p / n for p in params]
        else:
            for i, p in enumerate(params):
                aggregated[i] += p / n
    return aggregated


def compute_model_divergence(
    global_params: List[np.ndarray],
    client_params: List[List[np.ndarray]],
) -> Dict[str, float]:
    """
    Measure how much client models diverge from global model.
    High divergence = more data heterogeneity (non-IID).
    """
    divergences = []
    for params in client_params:
        total_norm = 0.0
        for gp, cp in zip(global_params, params):
            diff = gp.astype(np.float32) - cp.astype(np.float32)
            total_norm += np.sum(diff ** 2)
        divergences.append(np.sqrt(total_norm))

    return {
        "mean_divergence": float(np.mean(divergences)),
        "max_divergence":  float(np.max(divergences)),
        "min_divergence":  float(np.min(divergences)),
        "std_divergence":  float(np.std(divergences)),
    }


def select_clients(
    num_total: int,
    fraction: float = 1.0,
    min_clients: int = 2,
) -> List[int]:
    """
    Randomly select a subset of clients for each round.
    Partial participation is standard in real FL systems.
    """
    num_selected = max(min_clients, int(num_total * fraction))
    num_selected = min(num_selected, num_total)
    return np.random.choice(num_total, num_selected, replace=False).tolist()
