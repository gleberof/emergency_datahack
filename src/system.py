import pytorch_lightning as pl
import torch

from src.utils.torch import BinaryFocalLossWithLogits, threshold_search


class LenaSystem(pl.LightningModule):
    def __init__(self, model, alpha=0.25, gamma=2, lr=3e-4, weight_decay=0.001):
        super().__init__()
        self.model = model

        self.save_hyperparameters({"alpha": alpha, "gamma": gamma, "lr": lr, "weight_decay": weight_decay})

        self.criterion = BinaryFocalLossWithLogits(alpha=self.hparams.alpha, gamma=self.hparams.gamma)

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

        self.log("Val/score", score)
        self.log("Val/thresh", thresh)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=0.001,
        )

        return optimizer
