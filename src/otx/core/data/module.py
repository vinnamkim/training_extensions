from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING

from datumaro import Dataset as DmDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from otx.core.types.task import OTXTaskType

from .factory import OTXDatasetFactory

if TYPE_CHECKING:
    from otx.core.config.data import (
        DataModuleConfig,
        SubsetConfig,
    )

    from .dataset.base import OTXDataset


class OTXDataModule(LightningDataModule):
    def __init__(self, task: OTXTaskType, config: DataModuleConfig):
        self.task = task
        self.config = config
        self.subsets: dict[str, OTXDataset] = {}
        self.prepare_data_per_node = True

    def prepare_data(self):
        self._prepare_data = True
        dataset = DmDataset.import_from(
            self.config.data_root,
            format=self.config.format,
        )
        self.subsets = {}

        for name, dm_subset in dataset.subsets().items():
            try:
                config = self._get_config(name)

                self.subsets[name] = OTXDatasetFactory.create(
                    task=self.task,
                    dm_subset=dm_subset,
                    config=config,
                )
                log.info(f"Add name: {name}, self.subsets: {self.subsets}")
            except KeyError:
                log.warning(f"{name} has no config. Skip it")

    def _get_config(self, subset: str) -> SubsetConfig:
        if (config := self.config.subsets.get(subset)) is None:
            raise KeyError(f"Config has no '{subset}' subset configuration")

        return config

    def _get_dataset(self, subset: str) -> OTXDataset:
        if (dataset := self.subsets.get(subset)) is None:
            raise KeyError(
                f"Dataset has no '{subset}'. Available subsets = {self.subsets.keys()}",
            )
        return dataset

    def train_dataloader(self):
        config, dataset = self._get_config("train"), self._get_dataset("train")

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    def val_dataloader(self):
        config, dataset = self._get_config("val"), self._get_dataset("val")

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    def test_dataloader(self):
        config, dataset = self._get_config("test"), self._get_dataset("test")

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    def setup(self, stage):
        pass

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass
