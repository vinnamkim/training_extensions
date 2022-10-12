from noisy_label.gen_config import (
    gen_noise_labels,
    get_cfg,
    save_anno,
    gen_subset_anno,
    read_train_anno,
    merge_anno,
    get_init_subset_ids,
    get_bbox_noise_size,
    get_cls_noise_size,
    get_size,
)
from noisy_label.run_al_scenario import train_al_scenario
from noisy_label.run_al_scenario import test_al_scenario
from tempfile import TemporaryDirectory
from noisy_label.extract_feat import get_feats, get_cand
from noisy_label.filtration import (
    correct,
    drop,
    nothing,
    get_noisy_label_cands,
    get_noisy_label_cands,
)
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", required=True, type=str, help="Dataset name")
parser.add_argument(
    "--root-dir", default="/mnt/ssd2/sc_datasets_det", type=str, help="Dataset name"
)
parser.add_argument("--method", required=True, type=str, help="Dataset name")
parser.add_argument("--n-reps", default=1, type=int, help="N repetitions")
parser.add_argument("--noise-rate", default=0.05, type=float, help="Noise rate")
parser.add_argument("--n-add", default=16, type=int, help="N add")
parser.add_argument("--n-cycles", default=2, type=int, help="N cycles")

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
    csv_dir = "csv_results"

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for seed in range(n_reps):
        full_anno = read_train_anno(dname, root_dir=root_dir)
        subset_ids = get_init_subset_ids(full_anno, n_add, seed=seed)
        anno = gen_subset_anno(dname, subset_ids, root_dir=root_dir)
        anno = gen_noise_labels(anno, seed=seed, noise_rate=noise_rate)

        results = []

        for cycle in range(1, n_cycles + 1):
            with TemporaryDirectory() as work_dir:
                print(f"Cycle={cycle}, work_dir={work_dir}")
                train_ann_file = save_anno(anno, work_dir)
                train_cfg = get_cfg(
                    dname, root_dir=root_dir, train_ann_file=train_ann_file
                )

                dataset_size = get_size(anno)
                n_bbox_noise = get_bbox_noise_size(anno)
                n_cls_noise = get_cls_noise_size(anno)

                train_al_scenario(train_cfg, work_dir)
                output = test_al_scenario(train_cfg, work_dir)
                del output["bbox_mAP_copypaste"]

                cand_size = max(10, int(dataset_size * noise_rate))
                cand_size1 = cand_size // 2
                cand_size2 = cand_size - cand_size1

                noisy_cands = get_noisy_label_cands("output")
                noisy_cand_ids = (
                    noisy_cands["bbox_cands"][-cand_size1:]
                    + noisy_cands["cls_cands"][-cand_size2:]
                )
                if method == "correct":
                    anno, n_fix_bbox, n_fix_cls = correct(anno, noisy_cand_ids)
                elif method == "drop":
                    anno, n_fix_bbox, n_fix_cls = drop(anno, noisy_cand_ids)
                elif method == "nothing":
                    anno, n_fix_bbox, n_fix_cls = nothing(anno, noisy_cand_ids)
                else:
                    raise NotImplementedError()

                if cycle + 1 == n_cycles:
                    print("Next cycle is the last cycle. Don't add new samples.")
                elif cycle == n_cycles:
                    print("This is the last cycle.")
                else:
                    feature_vectors = get_feats(work_dir, dname)
                    seen_ids = [img["uid"] for img in anno["images"]]
                    cand_ids, scores = get_cand(seen_ids, feature_vectors, n_add)
                    anno_to_add = gen_subset_anno(dname, cand_ids, root_dir=root_dir)
                    anno_to_add = gen_noise_labels(
                        anno_to_add, seed=seed, noise_rate=noise_rate
                    )
                    anno = merge_anno(anno, anno_to_add)

                to_log = {}
                to_log["seed"] = seed
                to_log["dataset_size"] = dataset_size
                to_log["cycle"] = cycle
                to_log["dataset_name"] = dname
                to_log["noise_rate"] = noise_rate
                to_log["n_bbox_noise"] = n_bbox_noise
                to_log["n_cls_noise"] = n_cls_noise
                to_log["n_fix_bbox"] = n_fix_bbox
                to_log["n_fix_cls"] = n_fix_cls
                to_log["method"] = method
                for k, v in output.items():
                    to_log[k] = v

                print("to_log")
                print(to_log)
                results += [to_log]

                df = pd.DataFrame(results)
                csv_path = os.path.join(csv_dir, dname + ".csv")
                df.to_csv(csv_path)
