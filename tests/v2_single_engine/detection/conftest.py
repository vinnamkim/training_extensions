import pytest
from otx.v2_single_engine.enums.task import OTXTask
from otx.v2_single_engine.config import DataModuleConfig
from otx.v2_single_engine.config.data_module import SubsetConfig
import os.path as osp
from otx.v2_single_engine.data_module import OTXDataModule


@pytest.fixture
def fxt_mmcv_det_transform_config(fxt_rtmdet_tiny_config):
    return fxt_rtmdet_tiny_config.train_pipeline


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
        task=OTXTask.DETECTION,
        config=config,
    )
    datamodule.prepare_data()
    return datamodule
