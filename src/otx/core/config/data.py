# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects for data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from otx.core.types.transformer_libs import TransformLibType


@dataclass
class SubsetConfig:
    """DTO for dataset subset configuration."""

    batch_size: int
    num_workers: int
    subset_name: str

    transform_lib_type: TransformLibType
    transforms: list[dict[str, Any]]


@dataclass
class DataModuleConfig:
    """DTO for data module configuration."""

    data_format: str
    data_root: str

    train_subset: SubsetConfig
    val_subset: SubsetConfig
    test_subset: SubsetConfig

    mem_cache_size: str = "1GB"