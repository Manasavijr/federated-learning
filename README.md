# Privacy-Preserving Federated Learning System

Production-grade federated learning system implementing FedAvg with differential privacy, non-IID data simulation, and centralized vs. federated accuracy comparison — built with PyTorch and Flower.

Directly motivated by Apple's on-device ML and privacy-first AI infrastructure.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   FL Server                           │
│  FedAvg Aggregation + DP Noise Injection             │
│  Privacy Accountant (RDP Moments Accountant)         │
└───────────────┬──────────────────────────────────────┘
                │  broadcast global model
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐  ┌────────┐  ┌────────┐
│Client 1│  │Client 2│  │Client N│   (up to 50 clients)
│MNIST   │  │MNIST   │  │MNIST   │
│IID or  │  │Non-IID │  │Non-IID │
│Non-IID │  │(Dir α) │  │(Dir α) │
└────────┘  └────────┘  └────────┘
    │  local gradients (clipped)
    └───────────────────────────►  FedAvg → DP Noise → Global Update
```

---

## Key Features

| Component | Implementation |
|---|---|
| **Federated Averaging** | Weighted FedAvg (McMahan et al. 2017) |
| **Differential Privacy** | Gaussian mechanism + RDP moments accountant |
| **Data Heterogeneity** | Dirichlet non-IID partitioning (α=0.1–10) |
| **Privacy Accounting** | (ε, δ)-DP guarantee tracking per round |
| **Comparison** | Federated vs. centralized accuracy curves |
| **Dashboard** | Live FastAPI dashboard with Chart.js plots |
| **CLI** | Full experiment runner with JSON output |

---

## Setup

```bash
cd federated-learning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run Dashboard

```bash
uvicorn api.main:app --reload --port 8082
# Open http://localhost:8082
```

Configure experiment in the UI:
- Clients, rounds, local epochs
- IID vs Non-IID (Dirichlet α)
- DP on/off + noise multiplier
- Watch live accuracy/loss curves

## Run CLI Experiments

```bash
# Federated (IID, no DP)
python experiments/run_experiment.py --mode federated --rounds 20 --clients 10

# Federated (non-IID, α=0.1, very heterogeneous)
python experiments/run_experiment.py --mode federated --rounds 20 --no-iid --alpha 0.1

# With differential privacy (ε ≈ 3.0 after 20 rounds)
python experiments/run_experiment.py --mode federated --dp --noise 1.0 --clip 1.0

# Full comparison: federated vs centralized
python experiments/run_experiment.py --mode compare --rounds 20 --clients 10
```

## Run Tests

```bash
pytest tests/ -v
```

---

## Results (Typical)

| Setting | Best Accuracy | Notes |
|---|---|---|
| Centralized | ~99.2% | Full data, no privacy |
| Federated IID | ~98.8% | Within 0.4% of centralized |
| Federated Non-IID (α=0.5) | ~97.5% | Moderate heterogeneity |
| Federated Non-IID (α=0.1) | ~94.0% | High heterogeneity |
| Federated + DP (σ=1.0) | ~96.5% | (ε≈3.2, δ=1e-5)-DP after 20 rounds |

---

## Privacy Budget Estimation

```
ε = RDP(α) + log(1/δ)/(α-1)

σ=0.5, 20 rounds → ε≈12  (weak privacy)
σ=1.0, 20 rounds → ε≈3.2 (moderate privacy)
σ=2.0, 20 rounds → ε≈0.8 (strong privacy, accuracy drops)
```

Use `/api/v1/privacy/estimate?noise_multiplier=1.0&num_clients=10&rounds=20` to estimate before running.
