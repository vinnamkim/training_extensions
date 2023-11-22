from __future__ import annotations

from typing import TYPE_CHECKING

from datumaro import DatasetSubset

from otx.core.types.task import OTXTaskType
from otx.core.types.transformer_libs import TransformLibType

from .dataset.base import OTXDataset

if TYPE_CHECKING:
    from otx.core.config.data import SubsetConfig

__all__ = ["TransformLibFactory", "OTXDatasetFactory"]


class TransformLibFactory:
    @classmethod
    def generate(cls, config: SubsetConfig):
        if config.transform_lib_type == TransformLibType.MMCV:
            from .transform_libs.mmcv import MMCVTransformLib

            return MMCVTransformLib.generate(config)

        if config.transform_lib_type == TransformLibType.MMDET:
            from .transform_libs.mmdet import MMDetTransformLib

            return MMDetTransformLib.generate(config)

        raise NotImplementedError(config.transform_lib_type)


class OTXDatasetFactory:
    @classmethod
    def create(
        cls,
        task: OTXTaskType,
        dm_subset: DatasetSubset,
        config: SubsetConfig,
    ) -> OTXDataset:
        transforms = TransformLibFactory.generate(config)

        if task == OTXTaskType.DETECTION:
            from .dataset.detection import OTXDetectionDataset

            return OTXDetectionDataset(dm_subset, transforms)

        raise NotImplementedError(task)
