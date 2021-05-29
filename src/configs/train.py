from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    name: str = "LenaTrans"
    version: Optional[str] = None
    gpus: int = 1
    batch_size: int = 128
    num_workers: int = 8
    rnn_units: int = 128
    top_classifier_units: int = 32
    alpha: float = 0.25
    gamma: float = 2
    lr: float = 3e-4
    weight_decay: float = 1e-3
