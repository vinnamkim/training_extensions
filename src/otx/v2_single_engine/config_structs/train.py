from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from otx.v2_single_engine.types.task import OTXTaskType

from .data_module import DataModuleConfig
from .model import ModelConfig


@dataclass
class BasicConfig:
    task: OTXTaskType

    work_dir: Path
    data_dir: Path
    log_dir: Path
    output_dir: Path


class TrainerConfig(DictConfig):
    default_root_dir: Path
    accelerator: str
    precision: int
    max_epochs: int
    min_epochs: int
    devices: int
    check_val_every_n_epoch: int
    deterministic: bool


@dataclass
class TrainConfig:
    base: BasicConfig
    data: DataModuleConfig
    trainer: dict
    model: ModelConfig
    recipe: str | None
    train: bool
    test: bool


def register_configs():
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
