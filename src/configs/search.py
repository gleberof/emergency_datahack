from dataclasses import dataclass

from src.configs.train import TrainBiTransConfig, TrainConfig


@dataclass
class SearchConfig:
    max_epochs: int = 5
    n_trials: int = 100
    study_name: str = "LenaTrans"

    train_config: TrainConfig = TrainConfig()


@dataclass
class SearchBiTransConfig:
    train: TrainBiTransConfig

    study_name: str = "LenaBiTrans"
    max_epochs: int = 5
    n_trials: int = 100
