from dataclasses import dataclass
from typing import Optional

from src.configs.model import BiTransModelConfig


@dataclass
class TrainConfig:
    name: str = "LenaTrans"
    version: Optional[str] = None
    gpus: int = 1
    batch_size: int = 128
    num_workers: int = 16
    rnn_units: int = 128
    top_classifier_units: int = 32
    alpha: float = 0.25
    gamma: float = 2
    lr: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 20
    patience: int = 20


@dataclass
class TrainBiTransConfig:
    model: BiTransModelConfig

    name: str = "LenaBiTrans"
    train_only: bool = False
    log_graph: bool = False
    version: Optional[str] = None
    gpus: int = 1
    gradient_clip_val: float = 100.0
    batch_size: int = 128
    num_workers: int = 16
    alpha: float = 0.25
    gamma: float = 2.5
    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 20
    patience: int = 20
