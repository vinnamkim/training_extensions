from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
from torch import Tensor
from torchvision import tv_tensors

from otx.core.types.task import OTXTaskType


@dataclass()
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


@dataclass(kw_only=True)
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

    task: OTXTaskType
    image: np.ndarray | tv_tensors.Image
    img_info: ImageInfo

    @property
    def image_type(self) -> ImageType:
        if isinstance(self.image, np.ndarray):
            return ImageType.NUMPY
        if isinstance(self.image, tv_tensors.Image):
            return ImageType.TV_IMAGE

        raise TypeError(self.image)


@dataclass(kw_only=True)
class OTXPredEntity(OTXDataEntity):
    score: np.ndarray | Tensor


@dataclass(kw_only=True)
class OTXBatchDataEntity:
    """Base Batch data entity for OTX

    This entity is the output of PyTorch DataLoader,
    which is the direct input of OTXModel.

    :param task: OTX task definition
    :param images: List of B numpy image tensors (C x H x W)
    :param imgs_info: Meta information for images
    """

    task: OTXTaskType
    batch_size: int
    images: list[tv_tensors.Image]
    imgs_info: list[ImageInfo]

    @classmethod
    def collate_fn(cls, entities: list[OTXDataEntity]) -> OTXBatchDataEntity:
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
            task=task,
            batch_size=batch_size,
            images=[entity.image for entity in entities],
            imgs_info=[entity.img_info for entity in entities],
        )


@dataclass(kw_only=True)
class OTXBatchPredEntity(OTXBatchDataEntity):
    scores: list[np.ndarray] | list[Tensor]


class OTXBatchLossEntity(dict[str, Tensor]):
    pass
