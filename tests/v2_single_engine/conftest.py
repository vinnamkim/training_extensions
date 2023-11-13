import pytest
import os.path as osp


@pytest.fixture(scope="session")
def fxt_asset_dir():
    return osp.join(osp.dirname(__file__), "..", "assets")


@pytest.fixture(scope="session")
def fxt_rtmdet_tiny_config(fxt_asset_dir):
    from mmengine.config import Config as MMConfig

    config_path = osp.join(
        fxt_asset_dir,
        "mmdet_configs",
        "rtmdet_tiny_8xb32-300e_coco.py",
    )

    return MMConfig.fromfile(config_path)
