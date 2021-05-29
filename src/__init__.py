from pathlib import Path

ROOT_DIR = Path(__file__).parents[0].parents[0]
DATA_DIR = ROOT_DIR / "data"
TRACK1_DIR = DATA_DIR / "track_1"
TRACK1_EXTRA_DIR = DATA_DIR / "1_track_extra_train"
LOGGING_DIR = DATA_DIR / "logs"
MODEL_CHECKPOINTS_DIR = DATA_DIR / "model-checkpoints"
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS_DIR / "model.ckpt"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
DEFAULT_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"
OPTUNA_LOCAL_DATABASE = f"sqlite:///{DATA_DIR / 'optuna.sqlite'}"
