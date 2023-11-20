from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
from datumaro import Bbox, DatasetSubset, Image
from torchvision import tv_tensors

from otx.v2_single_engine.data_entity.base import ImageInfo, ImageType
from otx.v2_single_engine.data_entity.detection import DetBatchDataEntity, DetDataEntity
from otx.v2_single_engine.types.task import OTXTaskType

from .base import OTXDataset


class OTXDetectionDataset(OTXDataset):
    def __init__(self, dm_subset: DatasetSubset, transforms: list[Callable]) -> None:
        super().__init__(dm_subset, transforms)

    def _get_item_impl(self, index: int) -> DetDataEntity | None:
        item = self.dm_subset.get(id=self.ids[index], subset=self.dm_subset.name)

        img = item.media_as(Image)
        img_data = img.data
        if img_data.shape[-1] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        img_shape = img.size

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )

        entity = DetDataEntity(
            task=OTXTaskType.DETECTION,
            image_type=ImageType.NUMPY,
            image=img_data,
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                pad_shape=img_shape,
                scale_factor=(1.0, 1.0),
            ),
            bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            labels=torch.as_tensor([ann.label for ann in bbox_anns]),
        )

        return self.apply_transforms(entity)

    @property
    def collate_fn(self):
        return DetBatchDataEntity.collate_fn
