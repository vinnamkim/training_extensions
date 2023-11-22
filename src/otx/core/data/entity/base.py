from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Dict, Type, TypeVar, List, Generic
import numpy as np
from torch import Tensor
from torchvision import tv_tensors

from otx.core.types.task import OTXTaskType


@dataclass
class ImageInfo:
    """Meta info for image

    :param img_id: Image id
    :param img_shape: Image shape after preprocessing
    :param ori_shape: Image shape right after loading it
    :param pad_shape: Image shape before padding
    :param scale_factor: Scale factor if the image is rescaled during preprocessing
    """

    img_idx: int
    img_shape: tuple[int, int]
    ori_shape: tuple[int, int]
    pad_shape: tuple[int, int]
    scale_factor: tuple[float, float]


class ImageType(IntEnum):
    NUMPY = auto()
    TV_IMAGE = auto()


T_OTXDataEntity = TypeVar(
    "T_OTXDataEntity",
    bound="OTXDataEntity",
)


@dataclass
class OTXDataEntity:
    """Base data entity for OTX.

    This entity is the output of each OTXDataset,
    which can be go through the input preprocessing tranforms.

    :param task: OTX task definition
    :param image: Image tensor which can have different type according to `image_type`
        1) `image_type=ImageType.NUMPY`: H x W x C numpy image tensor
        2) `image_type=ImageType.TV_IMAGE`: C x H x W torchvision image tensor
    :param imgs_info: Meta information for images
    """

    image: np.ndarray | tv_tensors.Image
    img_info: ImageInfo

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition"""
        raise RuntimeError("OTXTaskType is not defined.")

    @property
    def image_type(self) -> ImageType:
        if isinstance(self.image, np.ndarray):
            return ImageType.NUMPY
        if isinstance(self.image, tv_tensors.Image):
            return ImageType.TV_IMAGE

        raise TypeError(self.image)


@dataclass
class OTXPredEntity(OTXDataEntity):
    score: np.ndarray | Tensor


T_OTXBatchDataEntity = TypeVar(
    "T_OTXBatchDataEntity",
    bound="OTXBatchDataEntity",
)


@dataclass
class OTXBatchDataEntity(Generic[T_OTXDataEntity]):
    """Base Batch data entity for OTX

    This entity is the output of PyTorch DataLoader,
    which is the direct input of OTXModel.

    :param images: List of B numpy image tensors (C x H x W)
    :param imgs_info: Meta information for images
    """

    batch_size: int
    images: list[tv_tensors.Image]
    imgs_info: list[ImageInfo]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition"""
        raise RuntimeError("OTXTaskType is not defined.")

    @classmethod
    def collate_fn(cls, entities: List[T_OTXDataEntity]) -> OTXBatchDataEntity:
        if (batch_size := len(entities)) == 0:
            raise RuntimeError("collate_fn() input should have > 0 entities")

        task = entities[0].task

        if not all(task == entity.task for entity in entities):
            raise RuntimeError("collate_fn() input should include a single OTX task")

        if not all(entity.image_type == ImageType.TV_IMAGE for entity in entities):
            raise RuntimeError(
                "All entities should be torchvision's Image tensor before collate_fn()",
            )

        return OTXBatchDataEntity(
            batch_size=batch_size,
            images=[entity.image for entity in entities],
            imgs_info=[entity.img_info for entity in entities],
        )


T_OTXBatchPredEntity = TypeVar(
    "T_OTXBatchPredEntity",
    bound="OTXBatchPredEntity",
)


@dataclass
class OTXBatchPredEntity(OTXBatchDataEntity):
    scores: list[np.ndarray] | list[Tensor]


T_OTXBatchLossEntity = TypeVar(
    "T_OTXBatchLossEntity",
    bound="OTXBatchLossEntity",
)


class OTXBatchLossEntity(Dict[str, Tensor]):
    pass
