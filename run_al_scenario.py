from noisy_label.gen_config import (
    gen_missing_labels,
    gen_noise_labels,
    get_cfg,
    get_miss_noise_size,
    save_anno,
    gen_subset_anno,
    read_train_anno,
    merge_anno,
    get_init_subset_ids,
    get_bbox_noise_size,
    get_cls_noise_size,
    get_size,
)
from noisy_label.run_al_scenario import pred_train_dataloader, train_al_scenario
from noisy_label.run_al_scenario import test_al_scenario
from tempfile import TemporaryDirectory
from noisy_label.extract_feat import get_feats, get_cand
from noisy_label.filtration import (
    correct,
    correct_missing_labels,
    drop,
    nothing,
    get_noisy_cands,
)
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", required=True, type=str, help="Dataset name")
parser.add_argument(
    "--root-dir",
    default="/local_ssd2/vinnamki/sc_datasets_det",
    type=str,
    help="Dataset name",
)
parser.add_argument("--method", required=True, type=str, help="Dataset name")
parser.add_argument("--n-reps", default=10, type=int, help="N repetitions")
parser.add_argument("--noise-rate", default=0.05, type=float, help="Noise rate")
parser.add_argument("--n-add", default=16, type=int, help="N add")
parser.add_argument("--n-cycles", default=5, type=int, help="N cycles")
parser.add_argument(
    "--missing",
    action="store_true",
    help="If true, run missing label scenario.",
)

args = parser.parse_args()


if __name__ == "__main__":
    print("Start")
    print(args)
    dname = args.dataset_name
    root_dir = args.root_dir

    n_reps = args.n_reps
    noise_rate = args.noise_rate
    n_add = args.n_add

    n_cycles = args.n_cycles
    method = args.method
    csv_dir = os.path.join("csv_results", f"n_add_{n_add}_noise_rate_{noise_rate}")

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    results = []

    for seed in range(n_reps):
        full_anno = read_train_anno(dname, root_dir=root_dir)
        subset_ids = get_init_subset_ids(full_anno, n_add, seed=seed)
        anno = gen_subset_anno(dname, subset_ids, root_dir=root_dir)

        if args.missing:
            anno = gen_missing_labels(anno, seed=seed, noise_rate=noise_rate)
        else:
            anno = gen_noise_labels(anno, seed=seed, noise_rate=noise_rate)

        for cycle in range(1, n_cycles + 1):
            with TemporaryDirectory() as work_dir:
                print(f"Cycle={cycle}, work_dir={work_dir}")
                train_ann_file = save_anno(anno, work_dir)
                train_cfg = get_cfg(
                    dname, root_dir=root_dir, train_ann_file=train_ann_file
                )

                n_imgs, n_anns = get_size(anno)
                n_bbox_noise = get_bbox_noise_size(anno)
                n_cls_noise = get_cls_noise_size(anno)
                n_miss_noise = get_miss_noise_size(anno)

                train_al_scenario(train_cfg, work_dir)
                output = test_al_scenario(train_cfg, work_dir)
                del output["bbox_mAP_copypaste"]
                train_preds = pred_train_dataloader(train_cfg, work_dir)

                if cycle == n_cycles:
                    print("This is the last cycle. Don't do anything for noisy labels.")
                    n_fix_bbox, n_fix_cls = 0, 0
                else:
                    if method == "correct":
                        noisy_cand_ids = get_noisy_cands(
                            10, n_anns, noise_rate, work_dir
                        )
                        anno, n_fix_bbox, n_fix_cls = correct(anno, noisy_cand_ids)
                        anno, n_fix_miss = correct_missing_labels(
                            anno,
                            train_preds,
                            noise_rate,
                            iou_thresholds=0.8,
                            min_size=10,
                        )
                    elif method == "drop":
                        noisy_cand_ids = get_noisy_cands(
                            2, n_anns, noise_rate, work_dir
                        )
                        anno, n_fix_bbox, n_fix_cls = drop(anno, noisy_cand_ids)
                    elif method == "nothing":
                        noisy_cand_ids = get_noisy_cands(
                            10, n_anns, noise_rate, work_dir
                        )
                        anno, n_fix_bbox, n_fix_cls, n_fix_miss = nothing(anno, noisy_cand_ids)
                    else:
                        raise NotImplementedError()

                if n_cycles - 1 <= cycle <= n_cycles:
                    add_n_bbox_noise = 0
                    add_n_cls_noise = 0
                    add_n_miss_noise = 0
                    print(
                        "Next cycle is the last cycle or this cycle is the last cycle. Don't add new samples."
                    )
                else:
                    feature_vectors = get_feats(work_dir, root_dir, dname)
                    seen_ids = [img["uid"] for img in anno["images"]]
                    cand_ids, scores = get_cand(seen_ids, feature_vectors, n_add)
                    anno_to_add = gen_subset_anno(dname, cand_ids, root_dir=root_dir)

                    if args.missing:
                        anno_to_add = gen_missing_labels(
                            anno_to_add, seed=seed, noise_rate=noise_rate
                        )
                    else:
                        anno_to_add = gen_noise_labels(
                            anno_to_add, seed=seed, noise_rate=noise_rate
                        )

                    add_n_bbox_noise = get_bbox_noise_size(anno_to_add)
                    add_n_cls_noise = get_cls_noise_size(anno_to_add)
                    add_n_miss_noise = get_miss_noise_size(anno_to_add)
                    anno = merge_anno(anno, anno_to_add)

                after_n_bbox_noise = get_bbox_noise_size(anno)
                after_n_cls_noise = get_cls_noise_size(anno)
                after_n_miss_noise = get_miss_noise_size(anno)

                to_log = {}
                to_log["seed"] = seed
                to_log["n_imgs"] = n_imgs
                to_log["n_anns"] = n_anns
                to_log["cycle"] = cycle
                to_log["dataset_name"] = dname
                to_log["noise_rate"] = noise_rate
                to_log["n_bbox_noise"] = n_bbox_noise
                to_log["n_cls_noise"] = n_cls_noise
                to_log["n_miss_noise"] = n_miss_noise
                to_log["after_n_bbox_noise"] = after_n_bbox_noise
                to_log["after_n_cls_noise"] = after_n_cls_noise
                to_log["after_n_miss_noise"] = after_n_miss_noise
                to_log["add_n_bbox_noise"] = add_n_bbox_noise
                to_log["add_n_cls_noise"] = add_n_cls_noise
                to_log["add_n_miss_noise"] = add_n_miss_noise
                to_log["n_fix_bbox"] = n_fix_bbox
                to_log["n_fix_cls"] = n_fix_cls
                to_log["n_fix_miss"] = n_fix_miss
                to_log["method"] = method
                for k, v in output.items():
                    to_log[k] = v

                print("to_log")
                print(to_log)
                results += [to_log]

                df = pd.DataFrame(results)
                csv_path = os.path.join(csv_dir, dname + "-" + method + ".csv")
                df.to_csv(csv_path)
