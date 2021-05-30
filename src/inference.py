import hydra
import pytorch_lightning as pl
import torch

from src.configs import register_configs
from src.configs.inference import InferenceConfig
from src.models import LenaBiTrans
from src.system import LenaSystem
from src.utils.torch import get_datamodule, get_embeddings_projections


def inference(cfg: InferenceConfig):
    datamodule = get_datamodule(batch_size=cfg.batch_size, num_workers=cfg.num_workers, train_only=False)

    # trainer
    trainer = pl.Trainer(
        benchmark=True,
        gpus=cfg.gpus if torch.cuda.is_available() else None,
        max_epochs=cfg.max_epochs
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

    system = LenaSystem.load_from_checkpoint(
        cfg.checkpoint_path,
        model=model,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        prediction_thresh=cfg.prediction_thresh,
    )

    predictions = trainer.predict(system, return_predictions=True, datamodule=datamodule)

    test = datamodule.test.copy()
    test["ice_jam"] = torch.cat(predictions).flatten().numpy()  # type: ignore
    test.to_csv(cfg.submission_path, index=False)


@hydra.main(config_path=None, config_name="inference")
def main(cfg: InferenceConfig):
    inference(cfg=cfg)


if __name__ == "__main__":
    register_configs()
    main()
