from typing import Literal
from otx.v2_single_engine.types.task import OTXTaskType
from dataclasses import dataclass
from omegaconf import DictConfig
from pathlib import Path
from .data_module import DataModuleConfig

# @dataclass
# class OptimizerConfig:
#     type: str
#     lr: float
#     weight_decay: float


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
class ModelConfig(DictConfig):
    _target_: str
    optimizer: dict
    scheduler: dict
    otx_model: dict
    compile: bool


@dataclass
class TrainConfig:
    base: BasicConfig
    data: DataModuleConfig
    trainer: dict
    model: ModelConfig
    recipe: str | None
    train: bool


def register_configs():
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=TrainConfig)
