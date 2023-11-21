from __future__ import annotations

from enum import Enum


class TransformLibType(str, Enum):
    TORCHVISION = "TORCHVISION"
    MMCV = "MMCV"
    MMDET = "MMDET"
