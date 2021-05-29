from pathlib import Path

ROOT_DIR = Path(__file__).parents[0].parents[0]
DATA_DIR = ROOT_DIR / "data"
TRACK1_DIR = DATA_DIR / "track_1"
TRACK1_EXTRA_DIR = DATA_DIR / "1_track_extra_train"
LOGGING_DIR = DATA_DIR / "logs"
MODEL_CHECKPOINTS_DIR = DATA_DIR / "model-checkpoints"
OPTUNA_LOCAL_DATABASE = f"sqlite:///{DATA_DIR / 'optuna.sqlite'}"
