from __future__ import annotations

from torchvision.transforms.v2 import Compose

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otx.v2_single_engine.config_structs.data_module import SubsetConfig


class TorchvisionTransformLib:
    @classmethod
    def generate(cls, config: SubsetConfig) -> Compose:
        raise NotImplementedError()
