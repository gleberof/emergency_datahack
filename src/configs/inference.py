from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class InferenceConfig:
    checkpoint_path: str = MISSING

    prediction_thresh: float = 0.23
    gpus: int = 1
    batch_size: int = 128
    num_workers: int = 16
    rnn_units: int = 128
    top_classifier_units: int = 64
    feat_trans_width: int = 64
    alpha: float = 0.25
    gamma: float = 2
    lr: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 20
    patience: int = 20
