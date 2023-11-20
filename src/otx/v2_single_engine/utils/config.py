from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING

from mmengine.config import Config as MMConfig
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def convert_conf_to_mmconfig_dict(cfg: DictConfig) -> MMConfig:
    dict_cfg = OmegaConf.to_container(cfg)

    def to_tuple(dict_: dict) -> dict:
        # MMDET Mosaic asserts whether "img_shape" is tuple
        # File "/home/vinnamki/miniconda3/envs/otxv2/lib/python3.10/site-packages/mmdet/datasets/transforms/transforms.py", line 2324, in __init__

        for k, v in dict_.items():
            if isinstance(v, list) and all(isinstance(elem, Number) for elem in v):
                dict_[k] = tuple(v)
            elif isinstance(v, dict):
                to_tuple(v)

        return dict_

    return MMConfig(cfg_dict=to_tuple(dict_cfg))
