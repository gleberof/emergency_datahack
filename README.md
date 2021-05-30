[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Emergeny Hack 2021 - Команда "Звездочка", 1й трек

https://emergencydatahack.ru

# Структура каталога
* `src` - код проекта
* `notebooks` - ноутбуки проекта (эксперименты и анализ данных)
* `data/track_1` - данные архива track_1 (необходимо добавить вручную)
* `Emergency Hack 2021.pdf` - презентация проекта

# Воспроизводимость
Следующие ноутбуки содержат обучение модели и генерацию файла с предсказаниями. Их можно запустить на [Google Colab](https://colab.research.google.com/). Ноутбук обучения содержит в себе код загрузки данных.
 - `notebooks/train.ipynb`
 - `notebooks/inference.ipynb`

# Установка
## При помощи `poetry`:
```bash
poetry install
pre-commit install
```

## При помощи Docker-Compose
Start the container:

```bash
docker-compose build star
```

# Использование
Инструкции написаны для запуска в `docker` контейнере. Для локального запуска команды остаются аналогичными.

Запустите контейнер:


```bash
docker-compose run star bash
```

Создайте таблицу с признаками:

```bash
python src/make_features_df.py
````

Обучите модель (этот шаг можно пропустить, если поместить уже имеющийся checkpoint модели в `data/model-checkpoints/model.ckpt`):

```bash
python src/train_bi_trans.py batch_size=8 model.top_classifier_units=32
```

Все опции конфигурации запуска скрипта обучения можно посмотреть в `src/config/train.py`:
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

Скопируйте лучший checkpoint в  `data/model-checkpoints/model.ckpt`.

Сгенерируйте предсказания:

```bash
# better to set lower num_workers in order to avoid resources issues
python src/inference.py batch_size=32 num_workers=4 submission_path=/emergency/data/submissions/docker_submission.csv
```

Файл с предсказаниями будет записан в to `data/submissions/docker_submission.csv`.
