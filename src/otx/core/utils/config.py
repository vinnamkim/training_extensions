from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Dict, List, Literal, Union

from mmengine.config import Config as MMConfig
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def to_tuple(dict_: dict) -> dict:
    # MMDET Mosaic asserts if "img_shape" is not tuple
    # File "/home/vinnamki/miniconda3/envs/otxv2/lib/python3.10/site-packages/mmdet/datasets/transforms/transforms.py", line 2324, in __init__

    for k, v in dict_.items():
        if isinstance(v, (tuple, list)) and all(isinstance(elem, Number) for elem in v):
            dict_[k] = tuple(v)
        elif isinstance(v, dict):
            to_tuple(v)

    return dict_


def to_list(dict_: dict) -> dict:
    # MMDET FPN asserts if "in_channels" is not list
    # File "/home/vinnamki/miniconda3/envs/otxv2/lib/python3.10/site-packages/mmdet/models/necks/fpn.py", line 88, in __init__

    for k, v in dict_.items():
        if isinstance(v, (tuple, list)) and all(isinstance(elem, Number) for elem in v):
            dict_[k] = list(v)
        elif isinstance(v, dict):
            to_list(v)

    return dict_


def convert_conf_to_mmconfig_dict(
    cfg: DictConfig, to: Literal["tuple", "list"] = "tuple",
) -> MMConfig:
    dict_cfg = OmegaConf.to_container(cfg)

    if to == "tuple":
        return MMConfig(cfg_dict=to_tuple(dict_cfg))
    if to == "list":
        return MMConfig(cfg_dict=to_list(dict_cfg))

    raise ValueError(to)


def mmconfig_dict_to_dict(obj: Union[MMConfig, List[MMConfig]]) -> Union[List, Dict]:
    if isinstance(obj, list):
        return [mmconfig_dict_to_dict(x) for x in obj]
    elif hasattr(obj, "to_dict"):
        return {k: mmconfig_dict_to_dict(v) for k, v in obj.to_dict().items()}

    return obj
