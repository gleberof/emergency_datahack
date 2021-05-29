from dataclasses import dataclass

from src.configs.train import TrainConfig


@dataclass
class SearchConfig:
    max_epochs: int = 5
    n_trials: int = 100
    study_name: str = "LenaTrans"

    train_config: TrainConfig = TrainConfig()
