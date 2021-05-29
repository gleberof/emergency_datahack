from dataclasses import dataclass

from src.configs.train_extra import TrainExtraConfig


@dataclass
class SearchExtraConfig:
    max_epochs: int = 5
    n_trials: int = 100
    study_name: str = "LenaTransExtra"

    train_config: TrainExtraConfig = TrainExtraConfig()
