from dataclasses import dataclass


@dataclass
class BiTransModelConfig:
    rnn_units: int = 128
    top_classifier_units: int = 64
    feat_trans_width: int = 64
