from __future__ import annotations

from typing import TYPE_CHECKING

from torchvision.transforms.v2 import Compose

if TYPE_CHECKING:
    from otx.core.config.data import SubsetConfig


class TorchvisionTransformLib:
    @classmethod
    def generate(cls, config: SubsetConfig) -> Compose:
        raise NotImplementedError
