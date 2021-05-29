from hydra.core.config_store import ConfigStore

from src.configs.train import TrainConfig


def register_configs():
    cs = ConfigStore.instance()

    cs.store(node=TrainConfig, name="train")
