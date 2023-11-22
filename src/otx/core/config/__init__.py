from dataclasses import dataclass, field
from typing import Optional
from .data import DataModuleConfig
from .model import ModelConfig
from .base import BaseConfig
from .trainer import TrainerConfig


@dataclass
class TrainConfig:
    base: BaseConfig
    data: DataModuleConfig
    trainer: TrainerConfig
    model: ModelConfig
    logger: dict
    recipe: Optional[str]
    train: bool
    test: bool
    callbacks: list = field(default_factory=list)


def register_configs() -> None:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
