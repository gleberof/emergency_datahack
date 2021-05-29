from hydra.core.config_store import ConfigStore

from src.configs.search import SearchConfig
from src.configs.search_extra import SearchExtraConfig
from src.configs.train import TrainConfig
from src.configs.train_extra import TrainExtraConfig


def register_configs():
    cs = ConfigStore.instance()

    cs.store(node=TrainConfig, name="train")
    cs.store(node=SearchConfig, name="search")
    cs.store(node=TrainConfig, name="search", group="train")
    cs.store(node=TrainExtraConfig, name="train_extra")
    cs.store(node=SearchExtraConfig, name="search_extra")
    cs.store(node=TrainExtraConfig, name="search_extra", group="train_extra")
