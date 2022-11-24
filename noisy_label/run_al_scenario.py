import json
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test, train_detector
from mmdet.datasets import build_dataset
from mmdet.datasets.builder import build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env

from noisy_label.extract_feat import load_best_ckpt


def is_epoch_based_runner(runner_config: Config):
    return "Epoch" in runner_config.type


def patch_adaptive_repeat_dataset(
    config: Config, num_samples: int, decay: float = -0.002, factor: float = 30
):
    """Patch the repeat times and training epochs adatively

    Frequent dataloading inits and evaluation slow down training when the
    sample size is small. Adjusting epoch and dataset repetition based on
    empirical exponential decay improves the training time by applying high
    repeat value to small sample size dataset and low repeat value to large
    sample.

    :param config: mmcv config
    :param num_samples: number of training samples
    :param decay: decaying rate
    :param factor: base repeat factor
    """
    data_train = config.data.train
    if data_train.type == "MultiImageMixDataset":
        data_train = data_train.dataset
    if data_train.type == "RepeatDataset" and getattr(
        data_train, "adaptive_repeat_times", False
    ):
        if is_epoch_based_runner(config.runner):
            cur_epoch = config.runner.max_epochs
            new_repeat = max(round(math.exp(decay * num_samples) * factor), 1)
            new_epoch = math.ceil(cur_epoch / new_repeat)
            if new_epoch == 1:
                return
            config.runner.max_epochs = new_epoch
            data_train.times = new_repeat


def train_al_scenario(cfg: Config, work_dir: str = "output"):
    from external.mmdetection.detection_tasks.extension.utils.hooks import (
        ReduceLROnPlateauLrUpdaterHook,
    )

    cfg.work_dir = work_dir

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )

    train_dataset = build_dataset(cfg.data.train)

    patch_adaptive_repeat_dataset(cfg, len(train_dataset))

    datasets = [build_dataset(cfg.data.train)]

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])

    # %%
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta={"env_info": env_info},
    )

    return model


def test_al_scenario(cfg: Config, work_dir: str = "output"):
    cfg.work_dir = work_dir

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )

    model = load_best_ckpt(model, output_dir=work_dir)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)

    return dataset.evaluate(outputs)


def pred_train_dataloader(cfg: Config, work_dir: str = "output"):
    cfg.work_dir = work_dir

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )

    model = load_best_ckpt(model, output_dir=work_dir)
    data_cfg = deepcopy(cfg.data.test)
    data_cfg["img_prefix"] = cfg.data.train["dataset"]["img_prefix"]
    data_cfg["ann_file"] = cfg.data.train["dataset"]["ann_file"]
    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)

    return outputs
