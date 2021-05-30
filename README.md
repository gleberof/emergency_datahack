# Emergeny Hack 2021 - Команда "Звездочка", 1й трек

https://emergencydatahack.ru

# Структура каталога
* src - код проекта
* notebooks - ноутбуки проекта (экспериметы и анализ данных)
* data - данные (не выгружены d htgjpbnjhbq)
* Emergency Hack 2021.pdf - презентация проекта


# Installation
```bash
poetry install
pre-commit install
```

Place the `track_1` directory at `data/track_1`.

# Usage
Start the container:

```bash
docker-compose build star
docker-compose run star bash
```

Create the features dataframe:

```bash
python src/make_features_df.py
````

Train the model (to skip this step create a `data/model-checkpoints/model.ckpt` file):

```bash
python src/train_bi_trans.py batch_size=8 model.top_classifier_units=32
```

Full configuration:
```python
@dataclass
class BiTransModelConfig:
    rnn_units: int = 128
    top_classifier_units: int = 64
    feat_trans_width: int = 64


@dataclass
class TrainBiTransConfig:
    model: BiTransModelConfig

    name: str = "LenaBiTrans"
    train_only: bool = False
    log_graph: bool = False
    version: Optional[str] = None
    gpus: int = 1
    batch_size: int = 128
    num_workers: int = 16
    alpha: float = 0.25
    gamma: float = 2
    lr: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 20
    patience: int = 20
```

Copy the best checkpoint to `data/model-checkpoints/model.ckpt`.

Run the inference:

```bash
# better to set lower num_workers in order to avoid resources issues
python src/inference.py batch_size=32 num_workers=4 submission_path=/emergency/data/submissions/docker_submission.csv
```

The submission file will be written to `data/submissions/docker_submission.csv`.
