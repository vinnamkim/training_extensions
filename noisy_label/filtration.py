import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from mmdet.ops.nms import nms
from pycocotools.mask import iou

from noisy_label.gen_config import _reid


def get_missing_bboxes(anno):
    missing_bboxes = defaultdict(dict)
    for ann in anno["annotations"]:
        if ann["noise"] != 3:
            continue
        img_id = ann["image_id"]
        x1, y1, w, h = ann["bbox"]
        missing_bboxes[img_id][ann["id"]] = [(x1, y1, x1 + w, y1 + h)]
    return missing_bboxes


def get_bboxes(anno):
    bboxes = defaultdict(dict)
    for ann in anno["annotations"]:
        if ann["noise"] == 3:
            continue
        img_id = ann["image_id"]
        x1, y1, w, h = ann["bbox"]
        bboxes[img_id][ann["id"]] = [(x1, y1, x1 + w, y1 + h)]
    return bboxes


def get_missing_label_cands(
    anno, preds, noise_rate, iou_thresholds, min_size, do_nms=True
):
    bboxes = get_bboxes(anno)

    confs = []
    dts = {}

    for img_id in bboxes:
        gt = np.asarray(list(bboxes[img_id].values()))
        gt = gt.reshape(-1, 4)
        dt = preds[img_id]
        dt = np.concatenate([p for p in dt if p.shape[0] > 0], axis=0)

        if len(dt) == 0 or len(gt) == 0:
            continue

        if do_nms:
            dt, _ = nms(dt[:, :4], dt[:, 4], iou_thresholds)

        ious = iou(gt, dt[:, :4], np.zeros([len(gt)]))
        ious = ious.max(axis=0)

        dt = dt[ious < iou_thresholds]

        for dt_id, conf in enumerate(dt[:, 4]):
            confs += [(img_id, dt_id, conf)]

        dts[img_id] = dt[:, :4]

    N = len(anno["annotations"])
    M = max(min_size, int(N * noise_rate))
    cands = sorted(confs, key=lambda x: -x[-1])[:M]
    return cands, dts


def correct_missing_labels(anno, preds, noise_rate, iou_thresholds, min_size: int = 10):
    cands, dts = get_missing_label_cands(
        anno, preds, noise_rate, iou_thresholds, min_size
    )

    def _correct(anno, cands, dts):
        missing_bboxes = get_missing_bboxes(anno)
        recovers = []

        for cand in cands:
            img_id, dt_id, conf = cand
            # print(cand)
            dt = dts[img_id][dt_id].reshape(1, -1)

            for ann_id, gt in missing_bboxes[img_id].items():
                gt = np.asarray(gt, dtype=np.float32)
                ious = iou(gt, dt, np.zeros([len(gt)]))
                ious = ious.item()

                if ious > iou_thresholds:
                    # print(ann_id, dt, gt, ious, img_id)
                    recovers += [ann_id]

        recovers = set(recovers)
        n_fix_miss = 0
        for ann in anno["annotations"]:
            if ann["id"] in recovers:
                assert ann["noise"] == 3
                ann["noise"] = 0
                print(f"correct id={ann['id']}")
                n_fix_miss += 1

        return anno, n_fix_miss

    return _correct(anno, cands, dts)


def get_noisy_cands(min_size, n_anns, noise_rate, work_dir):
    cand_size = max(min_size, int(n_anns * noise_rate))
    cand_size1 = cand_size // 2
    cand_size2 = cand_size - cand_size1

    noisy_cands = _get_noisy_label_cands(work_dir)
    ids = (
        noisy_cands["bbox_cands"][-cand_size1:] + noisy_cands["cls_cands"][-cand_size2:]
    )
    return ids


def get_ema_loss_dyns(work_dir, gamma: float = 0.99):
    fpath = os.path.join(work_dir, "latest.pth")
    ckpt = torch.load(fpath)
    loss_dyns = ckpt["meta"]["loss_dynamics"]

    def _get_ema(key):
        out = {}
        for k, v in loss_dyns[key].items():
            dyns = [0.0] + v  # Start from zero to prevent bias to the initial value
            series = pd.Series(dyns)
            ema_v = series.ewm(alpha=1 - gamma).mean().iloc[-1]
            out[k] = ema_v
        return out

    return {key: _get_ema(key) for key in ["loss_dynamics_bbox", "loss_dynamics_cls"]}


def _get_noisy_label_cands(work_dir, gamma: float = 0.99):
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
    fix_miss_cnt = 0
    cand_ids = set(cand_ids)
    for ann in anno["annotations"]:
        if ann["id"] in cand_ids and ann["noise"] > 0:
            if ann["noise"] == 1:
                fix_bbox_cnt += 1
            elif ann["noise"] == 2:
                fix_cls_cnt += 1
            elif ann["noise"] == 3:
                fix_miss_cnt += 1

            print(f'Found {ann["id"]} type {ann["noise"]}')

    return anno, fix_bbox_cnt, fix_cls_cnt, fix_miss_cnt
