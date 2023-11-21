from otx.core.types.task import OTXTaskType
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BaseConfig:
    task: OTXTaskType

    work_dir: Path
    data_dir: Path
    log_dir: Path
    output_dir: Path
