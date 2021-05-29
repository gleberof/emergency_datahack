from hydra.core.config_store import ConfigStore

from src.configs.hydra import HydraConfig
from src.configs.train import TrainConfig


def register_configs():
    cs = ConfigStore.instance()

    cs.store(node=HydraConfig, name="hydra")
    cs.store(node=TrainConfig, name="train")
