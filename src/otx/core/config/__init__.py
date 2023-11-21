from dataclasses import dataclass
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
    recipe: str | None
    train: bool
    test: bool


def register_configs():
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
