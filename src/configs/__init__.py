from hydra.core.config_store import ConfigStore

from src.configs.inference import InferenceConfig
from src.configs.model import BiTransModelConfig
from src.configs.search import SearchBiTransConfig, SearchConfig
from src.configs.search_extra import SearchExtraConfig
from src.configs.train import TrainBiTransConfig, TrainConfig
from src.configs.train_extra import TrainExtraConfig


def register_configs():
    cs = ConfigStore.instance()

    cs.store(node=InferenceConfig, name="inference")
    cs.store(node=BiTransModelConfig, group="inference", name="model")
    cs.store(node=TrainConfig, name="train")
    cs.store(node=TrainBiTransConfig, name="train_bi_trans")
    cs.store(node=SearchConfig, name="search")
    cs.store(node=SearchBiTransConfig, name="search_bi_trans")
    cs.store(node=TrainConfig, name="search", group="train")
    cs.store(node=TrainBiTransConfig, name="search_bi_trans", group="train_bi_trans")
    cs.store(node=TrainExtraConfig, name="train_extra")
    cs.store(node=SearchExtraConfig, name="search_extra")
    cs.store(node=TrainExtraConfig, name="search_extra", group="train_extra")
