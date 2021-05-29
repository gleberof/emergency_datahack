from dataclasses import dataclass

from src.configs.train import TrainConfig


@dataclass
class SearchConfig:
    max_epochs: int = 5
    n_trials: int = 100

    train_config: TrainConfig = TrainConfig()
