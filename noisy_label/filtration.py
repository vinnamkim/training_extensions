import os
from typing import List
import torch
import pandas as pd

from noisy_label.gen_config import _reid


def get_ema_loss_dyns(work_dir, gamma: float = 0.99):
    fpath = os.path.join(work_dir, "latest.pth")
    ckpt = torch.load(fpath)
    loss_dyns = ckpt["meta"]["loss_dynamics"]

    def _get_ema(key):
        out = {}
        for k, v in loss_dyns[key].items():
            series = pd.Series(v)
            ema_v = series.ewm(alpha=1 - gamma).mean().iloc[-1]
            out[k] = ema_v
        return out

    return {key: _get_ema(key) for key in ["loss_dynamics_bbox", "loss_dynamics_cls"]}


def get_noisy_label_cands(work_dir, gamma: float = 0.99):
    """
    Annotation ids sorted by the assending order of EMA loss dyns
    """
    ema_loss_dyns = get_ema_loss_dyns(work_dir, gamma)

    results = {}
    results["bbox_cands"] = [
        k
        for k, _ in sorted(
            ema_loss_dyns["loss_dynamics_bbox"].items(), key=lambda item: item[1]
        )
    ]
    results["cls_cands"] = [
        k
        for k, _ in sorted(
            ema_loss_dyns["loss_dynamics_cls"].items(), key=lambda item: item[1]
        )
    ]

    return results


def correct(anno, cand_ids: List[int]):
    fix_bbox_cnt = 0
    fix_cls_cnt = 0
    cand_ids = set(cand_ids)
    for ann in anno["annotations"]:
        if ann["id"] in cand_ids and ann["noise"] > 0:
            if ann["noise"] == 1:
                ann["bbox"] = ann["orig_bbox"]
                fix_bbox_cnt += 1
            elif ann["noise"] == 2:
                ann["category_id"] = ann["orig_category_id"]
                fix_cls_cnt += 1

            print(f'Correct {ann["id"]} type {ann["noise"]}')
            ann["noise"] = 0

    return anno, fix_bbox_cnt, fix_cls_cnt


def drop(anno, cand_ids: List[int]):
    fix_bbox_cnt = 0
    fix_cls_cnt = 0
    cand_ids = set(cand_ids)

    new_annotations = []
    for ann in anno["annotations"]:
        if ann["id"] in cand_ids and ann["noise"] > 0:
            print(f'Drop {ann["id"]} type {ann["noise"]}')

            if ann["noise"] == 1:
                fix_bbox_cnt += 1
            else:
                fix_cls_cnt += 1
        else:
            new_annotations += [ann]
    anno["annotations"] = _reid(new_annotations)

    return anno, fix_bbox_cnt, fix_cls_cnt


def nothing(anno, cand_ids: List[int]):
    fix_bbox_cnt = 0
    fix_cls_cnt = 0
    cand_ids = set(cand_ids)
    for ann in anno["annotations"]:
        if ann["id"] in cand_ids and ann["noise"] > 0:
            if ann["noise"] == 1:
                fix_bbox_cnt += 1
            elif ann["noise"] == 2:
                fix_cls_cnt += 1

            print(f'Found {ann["id"]} type {ann["noise"]}')

    return anno, fix_bbox_cnt, fix_cls_cnt
