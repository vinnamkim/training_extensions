# We have two nice abstracts: 1) Task and 2) Data entity (Datumaro annotations)
# Since the task determines the input and output format of the model, we can define the following abstract model class for each task.
# At this time, what OTX model developers need to do is very clear: Implement the two abstract functions.
from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING

from omegaconf import OmegaConf, DictConfig

# This is an example for MMDetection models
# In this way, we can easily import some models developed from the MM community
from .base import OTXDetectionModel, OTXDetectionLitModule
from mmengine.config import Config as MMConfig

if TYPE_CHECKING:
    from torch import nn
    from otx.v2_single_engine.data_entity.detection import DetBatchEntity
    from mmdet.models.data_preprocessors import DetDataPreprocessor


def dict_config_to_mmconfig(config: DictConfig):
    return MMConfig(cfg_dict=OmegaConf.to_container(config))


class MMDetCompatibleModel(OTXDetectionModel):
    def __init__(self, config: MMConfig | DictConfig):
        self.config = (
            dict_config_to_mmconfig(config)
            if isinstance(config, DictConfig)
            else config
        )
        super().__init__()

    def create_model(self) -> nn.Module:
        # import mmdet.models as _
        from mmdet.registry import MODELS
        from mmengine.registry import MODELS as MMENGINE_MODELS

        # RTMDet-Tiny has bug if we pass dictionary data_preprocessor to MODELS.build
        # We should inject DetDataPreprocessor to MMENGINE MODELS explicitly.
        det = MODELS.get("DetDataPreprocessor")
        MMENGINE_MODELS.register_module(module=det)

        return MODELS.build(self.config)

    def customize_inputs(self, entity: DetBatchEntity) -> Dict[str, Any]:
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        mmdet_inputs = {}

        mmdet_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmdet_inputs["data_samples"] = [
            DetDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_instances=InstanceData(
                    bboxes=bboxes,
                    labels=labels,
                ),
            )
            for img_info, bboxes, labels in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
            )
        ]
        preprocessor: DetDataPreprocessor = self.model.data_preprocessor
        mmdet_inputs = preprocessor(data=mmdet_inputs, training=self.training)

        mmdet_inputs["mode"] = "loss" if self.training else "predict"

        return mmdet_inputs


class MMDetCompatibleLitModule(OTXDetectionLitModule):
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

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )


# Those designs require OTX to have only one data pipeline and engine
