"""
Federated Learning Server — orchestrates training across clients.

Implements the full FedAvg algorithm:
  1. Initialize global model
  2. For each round:
     a. Select subset of clients
     b. Broadcast global model
     c. Clients train locally (E epochs)
     d. Collect updates
     e. FedAvg aggregation (+ optional DP noise)
     f. Evaluate global model
     g. Log metrics
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from fl.clients.model import MNISTNet, get_model
from fl.clients.trainer import evaluate_global, train_local
from fl.data.dataset import (get_class_distribution, get_client_loaders,
                              get_test_loader, load_mnist, partition_iid,
                              partition_non_iid)
from fl.privacy.dp import (DPConfig, PrivacyAccountant, add_gaussian_noise)
from fl.server.aggregation import (compute_model_divergence, federated_average,
                                   select_clients)

logger = logging.getLogger(__name__)


@dataclass
class FLConfig:
    """Full federated learning configuration."""
    # FL params
    num_clients: int = 10
    num_rounds: int = 20
    fraction_fit: float = 1.0        # fraction of clients per round
    local_epochs: int = 2
    local_lr: float = 0.01
    batch_size: int = 32

    # Data
    iid: bool = True                 # IID vs non-IID partitioning
    dirichlet_alpha: float = 0.5     # non-IID heterogeneity (lower = more heterogeneous)
    data_dir: str = "./data/raw"

    # Differential privacy
    dp: DPConfig = field(default_factory=DPConfig)

    # Device
    device: str = "cpu"

    # Experiment name
    name: str = "fl_experiment"


@dataclass
class RoundResult:
    round_num: int
    test_accuracy: float
    test_loss: float
    num_clients_participated: int
    client_metrics: List[Dict]
    model_divergence: Dict
    privacy_budget: Optional[Dict]
    duration_s: float


class FederatedServer:
    """
    Coordinates the federated learning process.
    Maintains global model state and aggregates client updates.
    """

    def __init__(self, config: FLConfig):
        self.config = config
        self.global_model = get_model()
        self.round_results: List[RoundResult] = []
        self.best_accuracy = 0.0
        self.best_round = 0

        # Privacy accountant
        self.privacy_accountant = None
        if config.dp.enabled:
            self.privacy_accountant = PrivacyAccountant(
                config=config.dp,
                num_clients=config.num_clients,
                dataset_size=60000,  # MNIST train size
            )

        # Load data
        logger.info("Loading MNIST dataset...")
        self.train_dataset, self.test_dataset = load_mnist(config.data_dir)
        self.test_loader = get_test_loader(self.test_dataset)

        # Partition data across clients
        logger.info(f"Partitioning data ({'IID' if config.iid else 'non-IID'}, {config.num_clients} clients)...")
        if config.iid:
            self.client_indices = partition_iid(self.train_dataset, config.num_clients)
        else:
            self.client_indices = partition_non_iid(
                self.train_dataset, config.num_clients, alpha=config.dirichlet_alpha
            )

        # Log client data distributions
        for i, indices in enumerate(self.client_indices):
            dist = get_class_distribution(self.train_dataset, indices)
            logger.debug(f"Client {i}: {len(indices)} samples, classes: {dist}")

        logger.info(f"Server initialized. {config.num_clients} clients, {config.num_rounds} rounds")
        logger.info(f"DP enabled: {config.dp.enabled}, noise_multiplier={config.dp.noise_multiplier}")

    def get_global_params(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.global_model.state_dict().values()]

    def set_global_params(self, params: List[np.ndarray]) -> None:
        from collections import OrderedDict
        state = OrderedDict({
            k: torch.tensor(v)
            for k, v in zip(self.global_model.state_dict().keys(), params)
        })
        self.global_model.load_state_dict(state)

    def train_round(
        self,
        round_num: int,
        progress_callback: Optional[Callable] = None,
    ) -> RoundResult:
        """Execute one federated learning round."""
        t0 = time.time()

        # Select clients
        selected = select_clients(
            self.config.num_clients,
            self.config.fraction_fit,
        )
        logger.info(f"Round {round_num}: selected clients {selected}")

        global_params = self.get_global_params()
        client_updates = []
        client_metrics = []
        client_params_list = []

        # Client local training
        for client_id in selected:
            train_loader, val_loader = get_client_loaders(
                self.train_dataset,
                self.client_indices[client_id],
                batch_size=self.config.batch_size,
            )

            # Create local model copy with global params
            local_model = get_model()
            local_model.set_parameters(global_params)

            # Train locally
            updated_params, metrics = train_local(
                local_model,
                train_loader,
                epochs=self.config.local_epochs,
                lr=self.config.local_lr,
                dp_config=self.config.dp if self.config.dp.enabled else None,
                device=self.config.device,
            )

            metrics["client_id"] = client_id
            metrics["num_samples"] = len(self.client_indices[client_id])
            client_updates.append((updated_params, metrics["samples"]))
            client_metrics.append(metrics)
            client_params_list.append(updated_params)

            if progress_callback:
                progress_callback({
                    "round": round_num,
                    "client_id": client_id,
                    "status": "trained",
                    "accuracy": metrics["accuracy"],
                })

        # FedAvg aggregation
        aggregated_params = federated_average(client_updates)

        # Differential privacy: add noise to aggregated update
        if self.config.dp.enabled:
            aggregated_params = add_gaussian_noise(
                aggregated_params,
                noise_multiplier=self.config.dp.noise_multiplier,
                max_grad_norm=self.config.dp.max_grad_norm,
                num_clients=len(selected),
            )

        # Update global model
        self.set_global_params(aggregated_params)

        # Evaluate global model
        eval_metrics = evaluate_global(self.global_model, self.test_loader, self.config.device)

        # Model divergence
        divergence = compute_model_divergence(global_params, client_params_list)

        # Privacy accounting
        privacy_budget = None
        if self.privacy_accountant:
            privacy_budget = self.privacy_accountant.update(round_num)

        duration = time.time() - t0

        if eval_metrics["test_accuracy"] > self.best_accuracy:
            self.best_accuracy = eval_metrics["test_accuracy"]
            self.best_round = round_num

        result = RoundResult(
            round_num=round_num,
            test_accuracy=eval_metrics["test_accuracy"],
            test_loss=eval_metrics["test_loss"],
            num_clients_participated=len(selected),
            client_metrics=client_metrics,
            model_divergence=divergence,
            privacy_budget=privacy_budget,
            duration_s=round(duration, 2),
        )
        self.round_results.append(result)

        logger.info(
            f"Round {round_num}: acc={eval_metrics['test_accuracy']:.4f}, "
            f"loss={eval_metrics['test_loss']:.4f}, "
            f"time={duration:.1f}s"
            + (f", ε={privacy_budget['epsilon']:.3f}" if privacy_budget else "")
        )

        return result

    def run(self, progress_callback: Optional[Callable] = None) -> List[RoundResult]:
        """Run all federated learning rounds."""
        logger.info(f"Starting federated training: {self.config.num_rounds} rounds")
        for r in range(1, self.config.num_rounds + 1):
            self.train_round(r, progress_callback)
            if self.privacy_accountant:
                acc = self.privacy_accountant
                if acc.epsilon_spent >= acc.config.epsilon_budget:
                    logger.warning(f"Privacy budget exhausted at round {r}! Stopping.")
                    break
        logger.info(f"Training complete. Best accuracy: {self.best_accuracy:.4f} at round {self.best_round}")
        return self.round_results

    def get_summary(self) -> Dict:
        if not self.round_results:
            return {}
        accuracies = [r.test_accuracy for r in self.round_results]
        losses = [r.test_loss for r in self.round_results]
        return {
            "experiment": self.config.name,
            "num_rounds_completed": len(self.round_results),
            "best_accuracy": self.best_accuracy,
            "best_round": self.best_round,
            "final_accuracy": accuracies[-1],
            "final_loss": losses[-1],
            "accuracy_curve": accuracies,
            "loss_curve": losses,
            "config": {
                "num_clients": self.config.num_clients,
                "local_epochs": self.config.local_epochs,
                "iid": self.config.iid,
                "dp_enabled": self.config.dp.enabled,
                "noise_multiplier": self.config.dp.noise_multiplier,
            },
            "privacy": {
                "epsilon_spent": self.privacy_accountant.epsilon_spent if self.privacy_accountant else None,
                "delta": self.config.dp.delta if self.config.dp.enabled else None,
            } if self.config.dp.enabled else None,
        }
