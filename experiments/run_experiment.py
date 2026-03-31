"""
Run federated learning experiments and compare with centralized baseline.

Usage:
    python experiments/run_experiment.py --mode federated --rounds 20 --clients 10
    python experiments/run_experiment.py --mode centralized --rounds 20
    python experiments/run_experiment.py --mode compare --rounds 20 --clients 10
    python experiments/run_experiment.py --mode federated --no-iid --dp --noise 1.0
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.clients.model import get_model, count_parameters
from fl.clients.trainer import evaluate_global
from fl.data.dataset import load_mnist, get_test_loader
from fl.privacy.dp import DPConfig
from fl.server.federated_server import FLConfig, FederatedServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_centralized(rounds: int, lr: float = 0.01, device: str = "cpu") -> dict:
    """Train centralized baseline on full MNIST dataset."""
    logger.info("=== Running Centralized Baseline ===")
    from torch.utils.data import DataLoader
    from fl.data.dataset import load_mnist, TRANSFORM
    from torchvision import datasets

    train_dataset, test_dataset = load_mnist()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = get_test_loader(test_dataset)

    model = get_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    accuracy_curve = []
    loss_curve = []

    for epoch in range(1, rounds + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        metrics = evaluate_global(model, test_loader, device)
        accuracy_curve.append(metrics["test_accuracy"])
        loss_curve.append(metrics["test_loss"])
        logger.info(f"Epoch {epoch}/{rounds}: acc={metrics['test_accuracy']:.4f}, loss={metrics['test_loss']:.4f}")

    return {
        "experiment": "centralized",
        "best_accuracy": max(accuracy_curve),
        "final_accuracy": accuracy_curve[-1],
        "accuracy_curve": accuracy_curve,
        "loss_curve": loss_curve,
        "num_rounds": rounds,
    }


def run_federated(args) -> dict:
    """Run federated learning experiment."""
    dp_config = DPConfig(
        enabled=args.dp,
        noise_multiplier=args.noise,
        max_grad_norm=args.clip,
        delta=1e-5,
    )

    config = FLConfig(
        num_clients=args.clients,
        num_rounds=args.rounds,
        fraction_fit=args.fraction,
        local_epochs=args.local_epochs,
        local_lr=args.lr,
        iid=not args.no_iid,
        dirichlet_alpha=args.alpha,
        dp=dp_config,
        device=args.device,
        name=f"federated_{'noniid' if args.no_iid else 'iid'}_{'dp' if args.dp else 'nodp'}",
    )

    logger.info(f"=== Running Federated Learning ===")
    logger.info(f"  Clients: {args.clients}, Rounds: {args.rounds}, IID: {not args.no_iid}")
    logger.info(f"  DP: {args.dp}, Noise: {args.noise if args.dp else 'N/A'}")
    logger.info(f"  Model parameters: {count_parameters(get_model()):,}")

    server = FederatedServer(config)
    server.run()
    return server.get_summary()


def save_results(results: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Experiments")
    parser.add_argument("--mode", choices=["federated", "centralized", "compare"], default="federated")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--no-iid", action="store_true", help="Non-IID data partitioning")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID")
    parser.add_argument("--dp", action="store_true", help="Enable differential privacy")
    parser.add_argument("--noise", type=float, default=1.0, help="DP noise multiplier")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    results = {}

    if args.mode in ("federated", "compare"):
        results["federated"] = run_federated(args)
        save_results(results["federated"], f"{args.output}/federated.json")

    if args.mode in ("centralized", "compare"):
        results["centralized"] = run_centralized(args.rounds, args.lr, args.device)
        save_results(results["centralized"], f"{args.output}/centralized.json")

    if args.mode == "compare":
        fed_acc = results["federated"]["best_accuracy"]
        cen_acc = results["centralized"]["best_accuracy"]
        gap = cen_acc - fed_acc
        comparison = {
            "federated_best_acc": fed_acc,
            "centralized_best_acc": cen_acc,
            "accuracy_gap": round(gap, 4),
            "gap_pct": f"{gap*100:.2f}%",
            "verdict": "Federated within 2% of centralized ✅" if gap < 0.02
                       else f"Federated {gap*100:.1f}% below centralized",
        }
        save_results(comparison, f"{args.output}/comparison.json")
        logger.info(f"\n{'='*50}")
        logger.info(f"COMPARISON RESULTS:")
        logger.info(f"  Federated best:   {fed_acc:.4f}")
        logger.info(f"  Centralized best: {cen_acc:.4f}")
        logger.info(f"  Gap: {gap*100:.2f}%")
        logger.info(f"{'='*50}")

    return results


if __name__ == "__main__":
    main()
