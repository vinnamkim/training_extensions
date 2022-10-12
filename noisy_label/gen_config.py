import json
import os
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
from mmcv import Config


def gen_noise_labels(anno, seed: int = 0, noise_rate: float = 0.05):
    random.seed(seed)
    annots = anno["annotations"]

    N = len(annots)
    M = max(int(N * noise_rate), 2)
    print(f"N={N}, M={M}")

    random.shuffle(annots)

    no_set = annots[:M]
    ok_set = annots[M:]

    images = {ann["id"]: ann for ann in anno["images"]}

    dxdys = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, -1])]
    dxdys = dxdys + [-dxdy for dxdy in dxdys]

    cat_ids = np.unique([ann["category_id"] for ann in annots])

    def _move_bbox(ann, delta=0.4):
        img_w, img_h = (
            images[ann["image_id"]]["width"],
            images[ann["image_id"]]["height"],
        )

        orig_bbox = deepcopy(ann["bbox"])
        x, y, w, h = orig_bbox
        # x1, y1, x2, y2 = x, y, x + w, y + h
        wh = np.array([w, h])

        dxdy = random.choice(dxdys)
        dxdy = delta * wh * dxdy

        x1y1 = np.array([x, y]) + dxdy
        x2y2 = x1y1 + wh

        x1, y1 = x1y1
        x2, y2 = x2y2

        x1 = np.clip(x1, 0, img_w)
        x2 = np.clip(x2, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        y2 = np.clip(y2, 0, img_h)

        ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        ann["noise"] = 1
        ann["orig_bbox"] = orig_bbox
        # assert x2 - x1 == w, f"x2={x2}, x1={x1}, w={w}"
        # assert y2 - y1 == h, f"y2={y2}, y1={y1}, h={h}"

    def _change_label(ann):
        _id = ann["category_id"]
        orig_id = deepcopy(ann["category_id"])
        new_id = random.choice(cat_ids)
        while _id == new_id:
            new_id = random.choice(cat_ids)

        ann["category_id"] = new_id.item()
        ann["noise"] = 2
        ann["orig_category_id"] = orig_id

    for idx, ann in enumerate(no_set):
        if idx < M // 2:
            _move_bbox(ann)
        else:
            _change_label(ann)

    for ann in ok_set:
        ann["noise"] = 0

    anno["annotations"] = no_set + ok_set

    return anno


def get_path_16_subset(
    dname, root_dir: str = "/mnt/ssd2/sc_datasets_det", seed: int = 1
):
    return os.path.join(
        root_dir, dname, "annotations", f"instances_train_16_{seed}.json"
    )


def save_anno(anno, work_dir):
    path = os.path.join(work_dir, "tmp.json")
    with open(path, "w") as fp:
        json.dump(anno, fp)
    return path


def read_train_anno(dname: str, root_dir: str = "/mnt/ssd2/sc_datasets_det"):
    ann_file = os.path.join(root_dir, dname, "annotations", "instances_train.json")

    with open(ann_file, "r", encoding="utf-8") as fp:
        anno = json.load(fp)

    N = len(anno["annotations"])
    for ann in anno["annotations"]:
        assert 0 <= ann["id"] < N

    N = len(anno["images"])
    for ann in anno["images"]:
        assert 0 <= ann["id"] < N

    return anno


def get_init_subset_ids(anno, num_samples: int, seed: int = 0):
    random.seed(seed)
    ids = [img["id"] for img in anno["images"]]
    random.shuffle(ids)
    return ids[:num_samples]


def get_size(anno) -> Tuple[int, int]:
    img_ids = [img["id"] for img in anno["images"]]
    ann_ids = [ann["id"] for ann in anno["annotations"]]
    return len(img_ids), len(ann_ids)


# def gen_noise_anno(
#     dname: str,
#     gen_ids: List[int],
#     existing_anno: Optional[Dict] = None,
#     root_dir: str = "/mnt/ssd2/sc_datasets_det",
# ):


def _set_uid(item):
    item["uid"] = item["id"]
    return item


def _reid(arr):
    idx = 0
    if isinstance(arr, dict):
        for k, v in arr.items():
            v["id"] = idx
            arr[k] = v
            idx += 1
        return arr
    elif isinstance(arr, list):
        new_arr = []
        for v in arr:
            v["id"] = idx
            idx += 1
            new_arr += [v]
        return new_arr
    raise ValueError()


def gen_subset_anno(
    dname: str,
    subset_ids: List[int],
    root_dir: str = "/mnt/ssd2/sc_datasets_det",
):
    anno = read_train_anno(dname, root_dir=root_dir)

    img_lookup = {img["id"]: img for img in anno["images"]}

    new_images = {img_id: _set_uid(img_lookup[img_id]) for img_id in subset_ids}
    new_images = {v["uid"]: v for _, v in new_images.items()}
    new_images = _reid(new_images)

    new_annotations = {}
    for annot in anno["annotations"]:
        img_uid = annot["image_id"]
        annot["img_uid"] = img_uid

        if annot["img_uid"] in new_images:
            img_uid = annot["img_uid"]
            annot["image_id"] = new_images[img_uid]["id"]
            new_annotations[annot["id"]] = annot
    new_annotations = _reid(new_annotations)

    anno["images"] = list(new_images.values())
    anno["annotations"] = list(new_annotations.values())

    return anno


def get_bbox_noise_size(anno):
    return len([ann for ann in anno["annotations"] if ann["noise"] == 1])


def get_cls_noise_size(anno):
    return len([ann for ann in anno["annotations"] if ann["noise"] == 1])


def merge_anno(anno1, anno2):
    for key in ["licenses", "info", "categories"]:
        assert anno1[key] == anno2[key]

    new_images = {}
    for v in anno1["images"]:
        new_images[v["uid"]] = deepcopy(v)
    for v in anno2["images"]:
        new_images[v["uid"]] = deepcopy(v)
    new_images = _reid(new_images)

    cnt = 0
    new_annotations = {}
    for annot in anno1["annotations"]:
        if annot["img_uid"] in new_images:
            img_uid = annot["img_uid"]
            new_annot = deepcopy(annot)
            new_annot["image_id"] = new_images[img_uid]["id"]
            new_annotations[cnt] = new_annot
            cnt += 1

    for annot in anno2["annotations"]:
        if annot["img_uid"] in new_images:
            img_uid = annot["img_uid"]
            new_annot = deepcopy(annot)
            new_annot["image_id"] = new_images[img_uid]["id"]
            new_annotations[cnt] = new_annot
            cnt += 1

    new_annotations = _reid(new_annotations)

    merged_anno = {key: anno1[key] for key in ["licenses", "info", "categories"]}

    merged_anno["images"] = [img for img in new_images.values()]
    merged_anno["annotations"] = [img for img in new_annotations.values()]

    return merged_anno


def get_cfg(
    dname,
    root_dir: str = "/mnt/ssd2/sc_datasets_det",
    config_path: str = "external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_ATSS/noise_config_16.py",
    train_ann_file: Optional[str] = None,
    use_small_val: bool = True
):
    ann_files = {
        key: os.path.join(root_dir, dname, "annotations", f"instances_{key}.json")
        for key in ["train", "val", "test"]
    }

    if train_ann_file:
        ann_files["train"] = train_ann_file

    small_val_path = os.path.join(root_dir, dname, "annotations", "instances_val_100.json")
    if use_small_val and os.path.exists(small_val_path):
        ann_files["val"] = small_val_path
        print("Use small val ann file")

    img_prefixes = {
        key: os.path.join(root_dir, dname, "images", key) for key in ann_files
    }

    print('ann_files["train"]', ann_files["train"])
    with open(ann_files["train"], "r", encoding="utf-8") as fp:
        anno = json.load(fp)

    annots = anno["annotations"]
    cnt_bbox_noise = 0
    cnt_cls_noise = 0
    for ann in annots:
        if "noise" in ann:
            key = ann["noise"]
            if key == 1:
                cnt_bbox_noise += 1
            elif key == 2:
                cnt_cls_noise += 1

    print(
        f"N={len(annots)} cnt_bbox_noise={cnt_bbox_noise} cnt_cls_noise={cnt_cls_noise}"
    )

    cfg = Config.fromfile(config_path)
    classes = [cat["name"] for cat in anno["categories"]]
    cfg.classes = classes
    cfg.model.bbox_head.num_classes = len(classes)
    cfg.data.train.dataset.classes = classes
    cfg.data.val.classes = classes
    cfg.data.test.classes = classes

    cfg.data.train.dataset.ann_file = ann_files["train"]
    cfg.data.val.ann_file = ann_files["val"]
    cfg.data.test.ann_file = ann_files["test"]

    cfg.data.train.dataset.img_prefix = img_prefixes["train"]
    cfg.data.val.img_prefix = img_prefixes["val"]
    cfg.data.test.img_prefix = img_prefixes["test"]

    cfg.gpu_ids = [0]
    cfg.seed = 0

    return cfg
