import hydra
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import LOGGING_DIR, MODEL_CHECKPOINTS_DIR
from src.configs import register_configs
from src.configs.train import TrainBiTransConfig
from src.models import LenaBiTrans
from src.system import LenaSystem
from src.utils.torch import get_datamodule, get_embeddings_projections


def train(cfg: TrainBiTransConfig, trial=None):
    logger = TensorBoardLogger(
        str(LOGGING_DIR),
        name=cfg.name,
        version=cfg.version,
        log_graph=cfg.log_graph,
        default_hp_metric=True,
    )

    checkpoints = ModelCheckpoint(
        dirpath=str(MODEL_CHECKPOINTS_DIR / cfg.name),
        monitor="hp_metric",
        verbose=True,
        mode="max",
        save_top_k=-1,
    )

    early_stopping = EarlyStopping(monitor="Val/f1_score", patience=cfg.patience)
    if trial:
        early_stopping = PyTorchLightningPruningCallback(monitor="Val/f1_score", trial=trial)  # type: ignore

    gpu_monitor = GPUStatsMonitor()

    datamodule = get_datamodule(batch_size=cfg.batch_size, num_workers=cfg.num_workers, train_only=cfg.train_only)

    # trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[gpu_monitor, checkpoints, early_stopping],
        profiler="simple",
        benchmark=True,
        gpus=cfg.gpus,
        max_epochs=cfg.max_epochs,
        val_check_interval=1 if not cfg.train_only else 1000,
        # enable_pl_optimizer=True,
    )

    embeddings_projections = get_embeddings_projections(
        categorical_features=datamodule.categorical_features, features_df=datamodule.features_df
    )

    model = LenaBiTrans(
        cat_features=datamodule.categorical_features,
        embeddings_projections=embeddings_projections,
        numerical_features=datamodule.numerical_features,
        station_col_name="hydro_fixed_station_id_categorical",
        day_col_name="day_target_categorical",
        rnn_units=cfg.model.rnn_units,
        top_classifier_units=cfg.model.top_classifier_units,
        feat_trans_width=cfg.model.feat_trans_width,
    )

    system = LenaSystem(model=model, alpha=cfg.alpha, gamma=cfg.gamma, lr=cfg.lr, weight_decay=cfg.weight_decay)

    trainer.fit(system, datamodule=datamodule)

    return trainer, system, datamodule


@hydra.main(config_path=None, config_name="train_bi_trans")
def main(cfg: TrainBiTransConfig):
    train(cfg=cfg)


if __name__ == "__main__":
    register_configs()
    main()
