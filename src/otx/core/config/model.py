from dataclasses import dataclass


@dataclass
class ModelConfig:
    _target_: str
    optimizer: dict
    scheduler: dict
    otx_model: dict
    compile: bool
