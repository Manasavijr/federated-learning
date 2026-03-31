"""FastAPI routes for federated learning experiments."""
import asyncio
import logging
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas.schemas import ExperimentConfig, ExperimentStatus, RoundMetrics
from fl.privacy.dp import DPConfig, compute_privacy_guarantee
from fl.server.federated_server import FLConfig, FederatedServer

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory experiment store
experiments: Dict[str, ExperimentStatus] = {}
servers: Dict[str, FederatedServer] = {}


async def run_experiment_bg(experiment_id: str, config: ExperimentConfig):
    """Run FL experiment in background."""
    status = experiments[experiment_id]
    status.status = "running"

    try:
        dp_config = DPConfig(
            enabled=config.dp_enabled,
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=config.max_grad_norm,
        )
        fl_config = FLConfig(
            num_clients=config.num_clients,
            num_rounds=config.num_rounds,
            local_epochs=config.local_epochs,
            local_lr=config.local_lr,
            fraction_fit=config.fraction_fit,
            iid=config.iid,
            dirichlet_alpha=config.dirichlet_alpha,
            dp=dp_config,
            name=config.experiment_name,
        )

        server = FederatedServer(fl_config)
        servers[experiment_id] = server

        for round_num in range(1, config.num_rounds + 1):
            result = server.train_round(round_num)
            status.current_round = round_num
            status.progress_pct = round(round_num / config.num_rounds * 100, 1)
            status.current_accuracy = result.test_accuracy
            status.metrics.append(RoundMetrics(
                round_num=result.round_num,
                test_accuracy=result.test_accuracy,
                test_loss=result.test_loss,
                num_clients=result.num_clients_participated,
                epsilon=result.privacy_budget["epsilon"] if result.privacy_budget else None,
                duration_s=result.duration_s,
            ))
            await asyncio.sleep(0)  # yield to event loop

        status.status = "completed"
        status.summary = server.get_summary()

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
        status.status = "failed"
        status.error = str(e)


@router.post("/experiments", response_model=ExperimentStatus)
async def create_experiment(config: ExperimentConfig, background_tasks: BackgroundTasks):
    """Start a new federated learning experiment."""
    experiment_id = str(uuid.uuid4())[:8]
    status = ExperimentStatus(
        experiment_id=experiment_id,
        status="pending",
        current_round=0,
        total_rounds=config.num_rounds,
        progress_pct=0.0,
        config=config.model_dump(),
    )
    experiments[experiment_id] = status
    background_tasks.add_task(run_experiment_bg, experiment_id, config)
    return status


@router.get("/experiments/{experiment_id}", response_model=ExperimentStatus)
async def get_experiment(experiment_id: str):
    """Get experiment status and metrics."""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiments[experiment_id]


@router.get("/experiments", response_model=list)
async def list_experiments():
    """List all experiments."""
    return [
        {"id": eid, "status": e.status, "accuracy": e.current_accuracy,
         "round": e.current_round, "total": e.total_rounds}
        for eid, e in experiments.items()
    ]


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Not found")
    experiments.pop(experiment_id)
    servers.pop(experiment_id, None)
    return {"deleted": experiment_id}


@router.post("/privacy/estimate")
async def estimate_privacy(
    noise_multiplier: float = 1.0,
    num_clients: int = 10,
    rounds: int = 20,
    delta: float = 1e-5,
):
    """Estimate (ε, δ) privacy guarantee before running experiment."""
    return compute_privacy_guarantee(noise_multiplier, num_clients, rounds, delta)
