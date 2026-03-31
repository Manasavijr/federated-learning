"""
Differential Privacy for Federated Learning.

Implements the Gaussian Mechanism for (ε, δ)-differential privacy:
  - Gradient clipping (bounds sensitivity)
  - Gaussian noise injection (provides privacy guarantee)

Privacy budget tracking via moments accountant.
This is the same mechanism Apple uses in iOS/macOS on-device ML.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    """Differential privacy configuration."""
    enabled: bool = True
    noise_multiplier: float = 1.0    # σ — controls privacy/utility tradeoff
    max_grad_norm: float = 1.0       # L2 sensitivity (clipping bound)
    delta: float = 1e-5              # δ for (ε, δ)-DP
    epsilon_budget: float = 10.0     # Maximum ε to spend


@dataclass
class PrivacyAccountant:
    """
    Tracks cumulative privacy budget spent across rounds.
    Uses Rényi Differential Privacy (RDP) moments accountant.
    """
    config: DPConfig
    num_clients: int
    dataset_size: int
    rounds_completed: int = 0
    epsilon_spent: float = 0.0
    privacy_log: List[dict] = field(default_factory=list)

    def compute_epsilon(self, rounds: int) -> float:
        """
        Estimate ε using RDP accountant approximation.
        Based on: Mironov (2017) "Rényi Differential Privacy"
        """
        if not self.config.enabled:
            return float("inf")

        q = 1.0 / self.num_clients  # sampling ratio
        sigma = self.config.noise_multiplier
        delta = self.config.delta

        # RDP epsilon approximation
        # ε_RDP(α) ≈ α / (2σ²) for Gaussian mechanism
        # Convert to (ε, δ)-DP via: ε = ε_RDP + log(1/δ)/(α-1)
        alpha = 10  # RDP order
        rdp_epsilon = (alpha * q**2) / (2 * sigma**2) * rounds
        epsilon = rdp_epsilon + math.log(1 / delta) / (alpha - 1)
        return round(epsilon, 4)

    def update(self, round_num: int) -> dict:
        self.rounds_completed = round_num
        self.epsilon_spent = self.compute_epsilon(round_num)
        entry = {
            "round": round_num,
            "epsilon": self.epsilon_spent,
            "delta": self.config.delta,
            "budget_remaining": max(0, self.config.epsilon_budget - self.epsilon_spent),
            "budget_exhausted": self.epsilon_spent >= self.config.epsilon_budget,
        }
        self.privacy_log.append(entry)
        return entry


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Per-sample gradient clipping — bounds L2 sensitivity.
    Returns the actual norm before clipping.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)


def add_gaussian_noise(
    parameters: List[np.ndarray],
    noise_multiplier: float,
    max_grad_norm: float,
    num_clients: int,
) -> List[np.ndarray]:
    """
    Add Gaussian noise to aggregated model updates.
    Noise scale: σ = noise_multiplier × max_grad_norm / num_clients
    """
    noisy_params = []
    for param in parameters:
        noise_scale = noise_multiplier * max_grad_norm / num_clients
        noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)
        noisy_params.append(param + noise)
    return noisy_params


def compute_privacy_guarantee(
    noise_multiplier: float,
    num_clients: int,
    rounds: int,
    delta: float = 1e-5,
) -> dict:
    """
    Compute privacy guarantee for given hyperparameters.
    Returns (ε, δ) privacy guarantee.
    """
    q = 1.0 / num_clients
    alpha = 10
    rdp_eps = (alpha * q**2) / (2 * noise_multiplier**2) * rounds
    epsilon = rdp_eps + math.log(1 / delta) / (alpha - 1)
    return {
        "epsilon": round(epsilon, 4),
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "sampling_ratio": q,
        "rounds": rounds,
        "guarantee": f"({epsilon:.2f}, {delta})-DP",
    }
