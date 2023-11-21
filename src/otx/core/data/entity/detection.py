from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import tv_tensors

from otx.core.types.task import OTXTaskType

from .base import OTXBatchDataEntity, OTXBatchPredEntity, OTXDataEntity, OTXPredEntity


@dataclass(kw_only=True)
class DetDataEntity(OTXDataEntity):
    """Data entity for detection task

    :param bboxes: Bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: Bbox labels as integer indices
    """

    task: OTXTaskType = OTXTaskType.DETECTION
    bboxes: tv_tensors.BoundingBoxes
    labels: torch.LongTensor


@dataclass(kw_only=True)
class DetPredEntity(DetDataEntity, OTXPredEntity):
    pass


@dataclass(kw_only=True)
class DetBatchDataEntity(OTXBatchDataEntity):
    """Data entity for detection task

    :param bboxes: A list of bbox annotations as top-left-bottom-right
        (x1, y1, x2, y2) format with absolute coordinate values
    :param labels: A list of bbox labels as integer indices
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[torch.LongTensor]

    @classmethod
    def collate_fn(cls, entities: list[DetDataEntity]) -> DetBatchDataEntity:
        batch_data = super().collate_fn(entities)
        return cls(
            task=batch_data.task,
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            bboxes=[entity.bboxes for entity in entities],
            labels=[entity.labels for entity in entities],
        )


@dataclass(kw_only=True)
class DetBatchPredEntity(DetBatchDataEntity, OTXBatchPredEntity):
    pass
