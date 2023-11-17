from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Callable

import numpy as np
from mmcv.transforms import LoadImageFromFile as MMCVLoadImageFromFile
from mmcv.transforms.builder import TRANSFORMS
from omegaconf import DictConfig, OmegaConf

from otx.v2_single_engine.data_entity.base import OTXDataEntity

if TYPE_CHECKING:
    from otx.v2_single_engine.config_structs.data_module import SubsetConfig


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
    def get_builder(cls):
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
            cls.get_builder().build(cls.to_mmconfig(cfg)) for cfg in config.transforms
        ]

        cls.check_mandatory_transforms(
            transforms,
            mandatory_transforms={LoadImageFromFile},
        )

        return transforms

    @staticmethod
    def to_mmconfig(cfg: DictConfig):
        mm_config = OmegaConf.to_container(cfg)

        def to_tuple(dict_: dict):
            # MMDET Mosaic asserts whether "img_shape" is tuple
            # File "/home/vinnamki/miniconda3/envs/otxv2/lib/python3.10/site-packages/mmdet/datasets/transforms/transforms.py", line 2324, in __init__

            for k, v in dict_.items():
                if isinstance(v, list) and all(isinstance(elem, Number) for elem in v):
                    dict_[k] = tuple(v)
                elif isinstance(v, dict):
                    to_tuple(v)

            return dict_

        return to_tuple(mm_config)
