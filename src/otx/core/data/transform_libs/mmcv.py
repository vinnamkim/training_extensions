from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from mmcv.transforms import LoadImageFromFile as MMCVLoadImageFromFile
from mmcv.transforms.builder import TRANSFORMS

from otx.core.data.entity.base import OTXDataEntity
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from otx.core.config.data import SubsetConfig
    from mmengine.registry import Registry


@TRANSFORMS.register_module(force=True)
class LoadImageFromFile(MMCVLoadImageFromFile):
    def transform(self, entity: OTXDataEntity) -> dict | None:
        img = entity.image

        if self.to_float32:
            img = img.astype(np.float32)

        results = {}
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]

        results["__otx__"] = entity

        return results


class MMCVTransformLib:
    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMCV"""
        return TRANSFORMS

    @classmethod
    def check_mandatory_transforms(
        cls,
        transforms: list[Callable],
        mandatory_transforms: set,
    ) -> None:
        for transform in transforms:
            t_transform = type(transform)
            mandatory_transforms.discard(t_transform)

        if len(mandatory_transforms) != 0:
            raise RuntimeError(f"{mandatory_transforms} should be included")

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        transforms = [
            cls.get_builder().build(convert_conf_to_mmconfig_dict(cfg))
            for cfg in config.transforms
        ]

        cls.check_mandatory_transforms(
            transforms,
            mandatory_transforms={LoadImageFromFile},
        )

        return transforms
