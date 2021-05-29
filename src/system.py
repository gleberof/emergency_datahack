from typing import Any, Optional

import joblib
import pytorch_lightning as pl
import torch

from src import DATA_DIR
from src.utils.torch import BinaryFocalLossWithLogits, threshold_search


class LenaSystem(pl.LightningModule):
    def __init__(self, model, alpha=0.25, gamma=2, lr=3e-4, weight_decay=0.001, prediction_thresh=None):
        super().__init__()
        self.model = model

        self.save_hyperparameters({"alpha": alpha, "gamma": gamma, "lr": lr, "weight_decay": weight_decay})

        self.criterion = BinaryFocalLossWithLogits(alpha=self.hparams.alpha, gamma=self.hparams.gamma)

        self.prediction_thresh = prediction_thresh

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        out = self(batch["x"].float(), batch["station"].long(), batch["day"].long())
        loss = self.criterion(out, batch["y"].float())  # .mean()

        self.log("Train/loss", loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["x"].float(), batch["station"].long(), batch["day"].long()).cpu()
        y = batch["y"].cpu()
        day = batch["day"].cpu()

        return {"out": out, "y": y, "day": day}

    def validation_epoch_end(self, outputs) -> None:
        y_proba = torch.sigmoid(torch.cat([o["out"] for o in outputs])).numpy()
        y_true = torch.cat([o["y"] for o in outputs]).numpy()

        thresh, score = threshold_search(y_proba=y_proba, y_true=y_true)

        self.log("Val/thresh", thresh)
        self.log("Val/f1_score", score)
        self.log("hp_metric", score)

    def test_step(self, batch, batch_idx):
        out = self(batch["x"].float(), batch["station"].long(), batch["day"].long()).cpu()
        y = batch["y"].cpu()
        day = batch["day"].cpu()

        return {"out": out, "y": y, "day": day}

    def test_epoch_end(self, outputs):
        y_proba = torch.sigmoid(torch.cat([o["out"] for o in outputs])).numpy()
        y_true = torch.cat([o["y"] for o in outputs]).numpy()

        thresh, score = threshold_search(y_proba=y_proba, y_true=y_true)

        self.log("Test/thresh", thresh)
        self.log("Test/f1_score", score)
        self.log("hp_metric", score)

        return score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=0.001,
        )

        return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        assert self.prediction_thresh is not None

        preds = (
            torch.sigmoid(self(batch["x"].float(), batch["station"].long(), batch["day"].long()))
            > self.prediction_thresh
        ).long()

        return preds


class LenaSystemExtra(pl.LightningModule):
    def __init__(self, model, alpha=0.25, gamma=2, lr=3e-4, weight_decay=0.001):
        super().__init__()
        self.model = model

        self.save_hyperparameters({"alpha": alpha, "gamma": gamma, "lr": lr, "weight_decay": weight_decay})

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.label_encoder = joblib.load(DATA_DIR / "dict_water_codes.joblib")
        self.target_code_index = self.label_encoder.classes_.tolist().index(12)

        self.class_weights = None

    def on_train_start(self) -> None:
        self.class_weights = torch.tensor(
            [
                1e-3 / (1e-3 + self.trainer.datamodule.train_ds.full_df[c].mean().item())  # type: ignore
                for c in self.trainer.datamodule.target_cols  # type: ignore
            ],
            device=self.device,
        )
        print(self.class_weights)
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=self.class_weights)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        out = self(batch["x"].float(), batch["station"].long(), batch["day"].long())
        loss = self.criterion(out, batch["y"].float())  # .mean()

        self.log("Train/loss", loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["x"].float(), batch["station"].long(), batch["day"].long()).cpu()
        y = batch["y"].cpu()
        day = batch["day"].cpu()

        return {"out": out, "y": y, "day": day}

    def validation_epoch_end(self, outputs) -> None:
        y_proba = (
            torch.sigmoid(torch.cat([o["out"] for o in outputs]))
            .flatten(start_dim=1)
            .numpy()[:, self.target_code_index]
        )

        y_true = torch.cat([o["y"] for o in outputs]).numpy()[:, self.target_code_index]

        thresh, score = threshold_search(y_proba=y_proba, y_true=y_true)

        self.log("Val/thresh", thresh)
        self.log("Val/f1_score", score)
        self.log("hp_metric", score)

    def test_step(self, batch, batch_idx):
        out = self(batch["x"].float(), batch["station"].long(), batch["day"].long()).cpu()
        y = batch["y"].cpu()
        day = batch["day"].cpu()

        return {"out": out, "y": y, "day": day}

    def test_epoch_end(self, outputs):
        y_proba = (
            torch.sigmoid(torch.cat([o["out"] for o in outputs]))
            .flatten(start_dim=1)
            .numpy()[:, self.target_code_index]
        )

        y_true = torch.cat([o["y"] for o in outputs]).numpy()[:, self.target_code_index]

        thresh, score = threshold_search(y_proba=y_proba, y_true=y_true)

        self.log("Test/thresh", thresh)
        self.log("Test/f1_score", score)
        self.log("hp_metric", score)

        return score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=0.001,
        )

        return optimizer
