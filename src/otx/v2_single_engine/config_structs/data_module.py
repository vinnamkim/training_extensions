from typing import Any

from dataclasses import dataclass

from otx.v2_single_engine.types.transformer_libs import TransformLibType


@dataclass
class SubsetConfig:
    batch_size: int
    num_workers: int

    transform_lib_type: TransformLibType
    transforms: list[dict[str, Any]]


@dataclass
class DataModuleConfig:
    format: str
    data_root: str
    subsets: dict[str, SubsetConfig]
