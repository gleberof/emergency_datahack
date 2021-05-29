import hydra

from src.configs import register_configs
from src.configs.inference import InferenceConfig


def inference(cfg: InferenceConfig):
    pass


@hydra.main(config_path=None, config_name="inference")
def main(cfg: InferenceConfig):
    inference(cfg=cfg)


if __name__ == "__main__":
    register_configs()
    main()
