import argparse

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import DATA_DIR, LOGGING_DIR, MODEL_CHECKPOINTS_DIR, TRACK1_DIR
from src.data import LenaDataModule
from src.models import LenaTrans
from src.system import LenaSystem
from src.utils.torch import get_embeddings_projections


def get_datamodule(batch_size, num_workers):
    train = pd.read_csv(TRACK1_DIR / "train.csv")
    test = pd.read_csv(TRACK1_DIR / "test.csv")
    features_df = pd.read_csv(DATA_DIR / "features.csv")
    datamodule = LenaDataModule(
        train=train, test=test, features_df=features_df, batch_size=batch_size, num_workers=num_workers
    )

    return datamodule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="LenaTrans")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--rnn-units", type=int, default=128)
    parser.add_argument("--top-classifier-units", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    logger = TensorBoardLogger(
        str(LOGGING_DIR),
        name=args.name,
        version=args.version,
        log_graph=False,
        default_hp_metric=True,
    )

    checkpoints = ModelCheckpoint(
        dirpath=str(MODEL_CHECKPOINTS_DIR / args.name),
        monitor="hp_metric",
        verbose=True,
        mode="max",
        save_top_k=-1,
    )

    gpu_monitor = GPUStatsMonitor()

    datamodule = get_datamodule(batch_size=args.batch_size, num_workers=args.num_workers)

    # trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[gpu_monitor, checkpoints],
        profiler="simple",
        benchmark=True,
        gpus=args.gpus
        # enable_pl_optimizer=True,
    )

    embeddings_projections = get_embeddings_projections(
        categorical_features=datamodule.categorical_features, features_df=datamodule.features_df
    )

    model = LenaTrans(
        cat_features=datamodule.categorical_features,
        embeddings_projections=embeddings_projections,
        numerical_features=datamodule.numerical_features,
        station_col_name="hydro_fixed_station_id_categorical",
        day_col_name="day_target_categorical",
        rnn_units=args.rnn_units,
        top_classifier_units=args.top_classifier_units,
    )

    system = LenaSystem(model=model, alpha=args.alpha, gamma=args.gamma, lr=args.lr, weight_decay=args.weight_decay)

    trainer.fit(system, datamodule=datamodule)
