from abc import abstractmethod
from collections.abc import Iterable
from typing import Callable

import numpy as np
from datumaro import DatasetSubset
from torch.utils.data import Dataset

from otx.v2_single_engine.data_entity.base import OTXDataEntity


class OTXDataset(Dataset):
    def __init__(
        self,
        dm_subset: DatasetSubset,
        transforms: Callable | list[Callable],
        max_refetch: int = 1000,
    ) -> None:
        self.dm_subset = dm_subset
        self.ids = [item.id for item in dm_subset]
        self.transforms = transforms
        self.max_refetch = max_refetch

    def __len__(self):
        return len(self.ids)

    def sample_another_idx(self) -> int:
        idx = np.random.randint(0, len(self))
        return idx

    def apply_transforms(self, entity: OTXDataEntity) -> OTXDataEntity | None:
        if callable(self.transforms):
            return self.transforms(entity)
        if isinstance(self.transforms, Iterable):
            return self._mmengine_transforms(entity)

        raise TypeError(self.transforms)

    def _mmengine_transforms(self, item: OTXDataEntity) -> OTXDataEntity | None:
        results = item
        for transform in self.transforms:
            results = transform(results)
            # MMCV transform can produce None. Please see
            # https://github.com/open-mmlab/mmengine/blob/26f22ed283ae4ac3a24b756809e5961efe6f9da8/mmengine/dataset/base_dataset.py#L59-L66
            if results is None:
                return None

        return results

    def __getitem__(self, index: int) -> OTXDataEntity:
        for _ in range(self.max_refetch):
            results = self._get_item_impl(index)

            if results is not None:
                return results

            index = self.sample_another_idx()

        raise RuntimeError(f"Reach the maximum refetch number ({self.max_refetch})")

    @abstractmethod
    def _get_item_impl(self, idx: int) -> OTXDataEntity:
        pass

    @property
    @abstractmethod
    def collate_fn(self):
        pass
