import pytest
from otx.v2_single_engine.types.task import OTXTaskType
from otx.v2_single_engine.config_structs.data_module import (
    DataModuleConfig,
    SubsetConfig,
)
import os.path as osp
from otx.v2_single_engine.data_module import OTXDataModule
from otx.v2_single_engine.utils.config import mmconfig_dict_to_dict
from omegaconf import DictConfig


@pytest.fixture
def fxt_mmcv_det_transform_config(fxt_rtmdet_tiny_config):
    return [
        DictConfig(cfg)
        for cfg in mmconfig_dict_to_dict(fxt_rtmdet_tiny_config.train_pipeline)
    ]


@pytest.fixture
def fxt_datamodule(fxt_asset_dir, fxt_mmcv_det_transform_config) -> OTXDataModule:
    data_root = osp.join(
        fxt_asset_dir,
        "car_tree_bug",
    )
    batch_size = 8
    num_workers = 0
    config = DataModuleConfig(
        format="coco_instances",
        data_root=data_root,
        subsets={
            "train": SubsetConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                transform_lib_type="MMDET",
                transforms=fxt_mmcv_det_transform_config,
            ),
            "val": SubsetConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                transform_lib_type="MMDET",
                transforms=fxt_mmcv_det_transform_config,
            ),
        },
    )
    datamodule = OTXDataModule(
        task=OTXTaskType.DETECTION,
        config=config,
    )
    datamodule.prepare_data()
    return datamodule
