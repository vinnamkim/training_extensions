# We have two nice abstracts: 1) Task and 2) Data entity (Datumaro annotations)
# Since the task determines the input and output format of the model, we can define the following abstract model class for each task.
# At this time, what OTX model developers need to do is very clear: Implement the two abstract functions.

from abc import abstractmethod
from typing import Any, Dict

import torch
from torch import Tensor, nn

from otx.v2_single_engine.data_entity.detection import DetBatchEntity
from otx.v2_single_engine.model.base import OTXModel, OTXLitModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class OTXDetectionModel(OTXModel):
    pass


class OTXDetectionLitModule(OTXLitModule):
    def __init__(
        self,
        otx_model: OTXDetectionModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        compile: bool,
    ):
        super().__init__(otx_model, optimizer, scheduler, compile)

        self.val_metric = MeanAveragePrecision()
        self.test_metric = MeanAveragePrecision()

    def on_validation_start(self) -> None:
        self.val_metric.reset()

    def on_test_start(self) -> None:
        self.test_metric.reset()

    def on_validation_end(self) -> None:
        self._log_metrics(self.val_metric, "val")

    def on_test_end(self) -> None:
        self._log_metrics(self.test_metric, "test")

    def _log_metrics(self, meter: MeanAveragePrecision, key: str):
        results = meter.compute()
        for k, v in results.items():
            self.log(
                f"{key}/{k}",
                v,
                sync_dist=True,
                prog_bar=True,
            )

    def validation_step(self, inputs: DetBatchEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
