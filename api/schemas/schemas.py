from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    num_clients: int = Field(default=10, ge=2, le=50)
    num_rounds: int = Field(default=10, ge=1, le=50)
    local_epochs: int = Field(default=2, ge=1, le=10)
    local_lr: float = Field(default=0.01, gt=0)
    fraction_fit: float = Field(default=1.0, gt=0, le=1.0)
    iid: bool = True
    dirichlet_alpha: float = Field(default=0.5, gt=0)
    dp_enabled: bool = False
    noise_multiplier: float = Field(default=1.0, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    experiment_name: str = "experiment"


class RoundMetrics(BaseModel):
    round_num: int
    test_accuracy: float
    test_loss: float
    num_clients: int
    epsilon: Optional[float] = None
    duration_s: float


class ExperimentStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    experiment_id: str
    status: str  # pending | running | completed | failed
    current_round: int
    total_rounds: int
    progress_pct: float
    current_accuracy: Optional[float] = None
    metrics: List[RoundMetrics] = []
    config: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
