import os
from typing import Dict, List

import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils.deployment import get_feature_vector
from sklearn.neighbors import NearestNeighbors

from .gen_config import get_cfg


def load_best_ckpt(model, output_dir):
    best_epoch = -1
    best_fpath = None

    for fname in os.listdir(output_dir):
        if "best" in fname:
            epoch = os.path.splitext(fname)[0].split("_")[-1]
            epoch = int(epoch)
            if epoch > best_epoch:
                best_epoch = epoch
                best_fpath = os.path.join(output_dir, fname)

    print(f"Load best epoch={best_epoch}")
    ckpt = torch.load(best_fpath)
    model.load_state_dict(ckpt["state_dict"])
    return model


def get_feats(output_dir, dname):
    cfg = get_cfg(
        dname,
        config_path="external/mmdetection/configs/custom-object-detection/gen3_mobilenetV2_ATSS/model.py",
    )
    cfg.data.test.ann_file = cfg.data.train.dataset.ann_file
    cfg.data.test.img_prefix = cfg.data.train.dataset.img_prefix

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model = load_best_ckpt(model, output_dir)

    cand_dataset = build_dataset(cfg.data.test)
    batch_size = 1
    cand_dataloader = build_dataloader(
        cand_dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )
    eval_model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    eval_model.eval()

    feature_vectors = []

    @torch.no_grad()
    def dump_features_hook(mod, inp, out):
        feature_vector = get_feature_vector(out)
        assert feature_vector.size(0) == 1
        feature_vectors.append(feature_vector.view(-1).detach().cpu().numpy())

    with eval_model.module.backbone.register_forward_hook(dump_features_hook):
        with torch.no_grad():
            for data in cand_dataloader:
                result = eval_model(return_loss=False, rescale=True, **data)

    feature_vectors = {idx: feat for idx, feat in enumerate(feature_vectors)}

    return feature_vectors


def compute_densities_knn(
    points: np.ndarray, compute_at_points: np.ndarray, n_neighbors: int
) -> np.ndarray:
    """
    Compute the density scores for a set of points based on a KNN density estimator
    initialized with another set of points.

    The point that is farthest from the KNN points will be assigned density 0.
    For the other points, the density will be scaled on proportion (linearly).

    :param points: MxD matrix of points to initialize KNN with. They determine density.
    :param compute_at_points: NxD matrix of points to compute the density for.
    :param n_neighbors: number of neighbors to use for KNN.
    :return: N-length array of density scores, in the same order of the relative points.
    """
    knn = NearestNeighbors(n_neighbors=min(len(points), n_neighbors))
    knn.fit(points)
    density_scores = []
    for i in range(compute_at_points.shape[0]):
        distances, _ = knn.kneighbors(compute_at_points[i : i + 1])
        distance_average = np.mean(distances)
        density_scores.append(distance_average)
    # normalize scores
    norm_density_scores = np.array(density_scores)
    norm_density_scores /= norm_density_scores.max()
    norm_density_scores = 1 - norm_density_scores
    return norm_density_scores


def compute_density_scores(
    embeddings_seen: np.ndarray,
    embeddings_unseen: np.ndarray,
) -> np.ndarray:
    """
    Compute the density scores for a set of points in the latent space.

    :param embeddings_seen: matrix (MxD) of stacked representation vectors computed on
        training dataset
    :param embeddings_unseen: matrix (NxD) of stacked representation vectors
        computed on unseen dataset
    :return: array of density scores of length N
    """
    num_seen = len(embeddings_seen)
    num_density_neighbors = max(1, min(num_seen, 5))
    density_scores = compute_densities_knn(
        points=embeddings_seen,
        compute_at_points=embeddings_unseen,
        n_neighbors=num_density_neighbors,
    )
    return density_scores


def get_cand(seen_ids: List[int], feature_vectors: Dict[int, np.ndarray], n_add: int):
    seen = seen_ids
    unseen = list(set(feature_vectors.keys()) - set(seen))

    seen_embeds = np.stack([feature_vectors[idx] for idx in seen])
    unseen_embeds = np.stack([feature_vectors[idx] for idx in unseen])

    scores = compute_density_scores(seen_embeds, unseen_embeds)

    cand_indices = scores.argsort()[:n_add]
    cand_ids = [unseen[idx] for idx in cand_indices]

    assert len(cand_ids) == n_add
    assert len(seen_ids) == len(set(seen_ids) - set(cand_ids))
    return cand_ids, scores
