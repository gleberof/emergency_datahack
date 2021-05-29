from dataclasses import dataclass

from src import DEFAULT_CHECKPOINT, DEFAULT_SUBMISSION_PATH
from src.configs.model import BiTransModelConfig


@dataclass
class InferenceConfig:

    model: BiTransModelConfig

    checkpoint_path: str = str(DEFAULT_CHECKPOINT)
    submission_path: str = str(DEFAULT_SUBMISSION_PATH)

    prediction_thresh: float = 0.23
    gpus: int = 1
    batch_size: int = 128
    num_workers: int = 16
    alpha: float = 0.25
    gamma: float = 2
    lr: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 20
    patience: int = 20
