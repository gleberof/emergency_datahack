import hydra
import pandas as pd
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import DATA_DIR, LOGGING_DIR, MODEL_CHECKPOINTS_DIR
from src.configs import register_configs
from src.configs.train_extra import TrainExtraConfig
from src.data import LenaDataModuleExtra
from src.models import LenaTransExtra
from src.system import LenaSystemExtra
from src.utils.torch import get_embeddings_projections


def get_datamodule(batch_size, num_workers):
    features_df = pd.read_csv(DATA_DIR / "features_extra.csv")
    datamodule = LenaDataModuleExtra(features_df=features_df, batch_size=batch_size, num_workers=num_workers)

    return datamodule


def train(cfg: TrainExtraConfig, trial=None):
    logger = TensorBoardLogger(
        str(LOGGING_DIR),
        name=cfg.name,
        version=cfg.version,
        log_graph=False,
        default_hp_metric=True,
    )

    checkpoints = ModelCheckpoint(
        dirpath=str(MODEL_CHECKPOINTS_DIR / cfg.name),
        monitor="hp_metric",
        verbose=True,
        mode="max",
        save_top_k=-1,
    )

    early_stopping = EarlyStopping(monitor="Val/f1_score")
    if trial:
        early_stopping = PyTorchLightningPruningCallback(monitor="Val/f1_score", trial=trial)  # type: ignore

    gpu_monitor = GPUStatsMonitor()

    datamodule = get_datamodule(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[gpu_monitor, checkpoints, early_stopping],
        profiler="simple",
        benchmark=True,
        gpus=cfg.gpus,
        max_epochs=cfg.max_epochs
        # enable_pl_optimizer=True,
    )

    embeddings_projections = get_embeddings_projections(
        categorical_features=datamodule.categorical_features, features_df=datamodule.features_df
    )

    model = LenaTransExtra(
        cat_features=datamodule.categorical_features,
        embeddings_projections=embeddings_projections,
        numerical_features=datamodule.numerical_features,
        target_cols=datamodule.target_cols,
        station_col_name="hydro_fixed_station_id_categorical",
        day_col_name="day_target_categorical",
        rnn_units=cfg.rnn_units,
        top_classifier_units=cfg.top_classifier_units,
    )

    system = LenaSystemExtra(model=model, alpha=cfg.alpha, gamma=cfg.gamma, lr=cfg.lr, weight_decay=cfg.weight_decay)

    trainer.fit(system, datamodule=datamodule)

    return trainer, system, datamodule


@hydra.main(config_path=None, config_name="train_extra")
def main(cfg: TrainExtraConfig):
    train(cfg=cfg)


if __name__ == "__main__":
    register_configs()
    main()
