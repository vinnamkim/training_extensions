from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig
from torchvision import tv_tensors

from otx.core.data.entity.detection import DetBatchPredEntity
from otx.core.model.entity.base import OTXModel
from otx.core.types.task import OTXTaskType
from otx.core.utils.config import convert_conf_to_mmconfig_dict


if TYPE_CHECKING:
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from torch import nn

    from otx.core.data.entity.detection import DetBatchDataEntity


class OTXDetectionModel(OTXModel):
    pass


# This is an example for MMDetection models
# In this way, we can easily import some models developed from the MM community
class MMDetCompatibleModel(OTXDetectionModel):
    def __init__(self, config: DictConfig):
        self.config = config
        super().__init__()

    def create_model(self) -> nn.Module:
        # import mmdet.models as _
        from mmdet.registry import MODELS
        from mmengine.registry import MODELS as MMENGINE_MODELS

        # RTMDet-Tiny has bug if we pass dictionary data_preprocessor to MODELS.build
        # We should inject DetDataPreprocessor to MMENGINE MODELS explicitly.
        det = MODELS.get("DetDataPreprocessor")
        MMENGINE_MODELS.register_module(module=det)

        try:
            model = MODELS.build(convert_conf_to_mmconfig_dict(self.config, to="tuple"))
        except AssertionError:
            model = MODELS.build(convert_conf_to_mmconfig_dict(self.config, to="list"))

        return model

    def customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
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

    def customize_outputs(self, outputs: Any) -> DetBatchPredEntity:
        from mmdet.structures import DetDataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            return {k: 1 / (len(v) + 1e-6) * sum(v) for k, v in outputs.items()}

        scores = []
        bboxes = []
        labels = []

        for output in outputs:
            if not isinstance(output, DetDataSample):
                raise TypeError(output)

            scores.append(output.pred_instances.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    output.pred_instances.bboxes,
                    format="XYXY",
                    canvas_size=output.img_shape,
                ),
            )
            labels.append(output.pred_instances.labels)

        return DetBatchPredEntity(
            task=OTXTaskType.DETECTION,
            batch_size=len(outputs),
            images=[],
            imgs_info=[],
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )
