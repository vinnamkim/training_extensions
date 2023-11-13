from __future__ import annotations

from typing import TYPE_CHECKING

from datumaro import DatasetSubset

from otx.v2_single_engine.enums.task import OTXTask
from otx.v2_single_engine.enums.transformers import TransformLibType

from .dataset.base import OTXDataset

if TYPE_CHECKING:
    from otx.v2_single_engine.config.data_module import SubsetConfig


class TransformerFactory:
    @classmethod
    def generate(cls, config: SubsetConfig):
        if config.transform_lib_type == TransformLibType.MMCV:
            from .transformers.mmcv import MMCVTransformLib

            return MMCVTransformLib.generate(config)

        if config.transform_lib_type == TransformLibType.MMDET:
            from .transformers.mmdet import MMDetTransformLib

            return MMDetTransformLib.generate(config)

        raise NotImplementedError(config.transform_lib_type)


class OTXDatasetFactory:
    @classmethod
    def create(
        cls,
        task: OTXTask,
        dm_subset: DatasetSubset,
        config: SubsetConfig,
    ) -> OTXDataset:
        transforms = TransformerFactory.generate(config)

        if task == OTXTask.DETECTION:
            from .dataset.detection import OTXDetectionDataset

            return OTXDetectionDataset(dm_subset, transforms)

        raise NotImplementedError(task)


__all__ = [TransformerFactory, OTXDatasetFactory]
