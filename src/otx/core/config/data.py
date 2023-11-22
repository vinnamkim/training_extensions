from dataclasses import dataclass
from typing import Any, List, Dict

from otx.core.types.transformer_libs import TransformLibType


@dataclass
class SubsetConfig:
    batch_size: int
    num_workers: int

    transform_lib_type: TransformLibType
    transforms: List[Dict[str, Any]]


@dataclass
class DataModuleConfig:
    format: str
    data_root: str
    subsets: Dict[str, SubsetConfig]
