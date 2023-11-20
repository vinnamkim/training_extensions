# We have two nice abstracts: 1) Task and 2) Data entity (Datumaro annotations)
# Since the task determines the input and output format of the model, we can define the following abstract model class for each task.
# At this time, what OTX model developers need to do is very clear: Implement the two abstract functions.

from abc import abstractmethod
from typing import Any, Dict

import torch
from lightning import LightningModule
from torch import Tensor, nn

from otx.v2_single_engine.data_entity.base import (
    OTXBatchDataEntity,
    OTXBatchLossEntity,
    OTXBatchPredEntity,
)


class OTXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.create_model()

    @abstractmethod
    def create_model(self) -> nn.Module:
        pass

    def customize_inputs(self, inputs: OTXBatchDataEntity) -> Dict[str, Any]:
        raise NotImplementedError

    def customize_outputs(self, outputs: Any) -> OTXBatchPredEntity:
        raise NotImplementedError

    def forward(
        self, inputs: OTXBatchDataEntity,
    ) -> OTXBatchLossEntity | OTXBatchPredEntity:
        # If customize_inputs is overrided
        outputs = (
            self.model(**self.customize_inputs(inputs))
            if self.customize_inputs != OTXModel.customize_inputs
            else self.model(inputs)
        )

        return (
            self.customize_outputs(outputs)
            if self.customize_outputs != OTXModel.customize_outputs
            else outputs
        )


class OTXLitModule(LightningModule):
    def __init__(
        self,
        otx_model: OTXModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        compile: bool,
    ):
        super().__init__()

        self.model = otx_model

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["otx_model"])

    def training_step(self, inputs: OTXBatchDataEntity, batch_idx: int) -> Tensor:
        train_loss = self.model(inputs)

        if isinstance(train_loss, Tensor):
            self.log(
                "train/loss", train_loss, on_step=True, on_epoch=False, prog_bar=True,
            )
            return train_loss
        elif isinstance(train_loss, dict):
            for k, v in train_loss.items():
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

            train_loss = sum(train_loss.values())
            self.log(
                "train/loss", train_loss, on_step=True, on_epoch=False, prog_bar=True,
            )
            return train_loss

        raise TypeError(train_loss)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.lr_scheduler_monitor_key,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @property
    def lr_scheduler_monitor_key(self) -> str:
        return "val/loss"
