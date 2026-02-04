from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .datasets import infer_2x2_pairs, subset_by_pair


@dataclass
class ClusteringResult:
    dataset: str
    model_name: str
    score: float
    std: float
    n_pairs: int



def clustering_score(cluster_assignments: np.ndarray, bio_labels: np.ndarray, center_labels: np.ndarray) -> float:
    ari_bio = adjusted_rand_score(bio_labels, cluster_assignments)
    ari_center = adjusted_rand_score(center_labels, cluster_assignments)
    return float(ari_bio - ari_center)



def _best_k_by_silhouette(features: np.ndarray, k_min: int, k_max: int, random_state: int) -> int:
    best_k = k_min
    best_s = -1.0
    for k in range(k_min, min(k_max, len(features) - 1) + 1):
        if k < 2:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        pred = km.fit_predict(features)
        if len(np.unique(pred)) < 2:
            continue
        s = silhouette_score(features, pred, metric="euclidean")
        if s > best_s:
            best_s = float(s)
            best_k = int(k)
    return best_k



def compute_clustering_score(
    dataset_name: str,
    model_name: str,
    features: np.ndarray,
    manifest_df: pd.DataFrame,
    repeats: int,
    k_min: int,
    k_max: int,
    max_pairs: Optional[int],
    random_state: int,
) -> ClusteringResult:
    pairs = infer_2x2_pairs(
        manifest_df,
        dataset_name=dataset_name,
        max_pairs=max_pairs,
        random_state=random_state,
    )
    if not pairs:
        raise RuntimeError(f"{dataset_name}: no valid 2x2 pairs for clustering")

    pair_scores = []

    for pidx, pair in enumerate(pairs):
        sub = subset_by_pair(manifest_df, pair)
        if len(sub) < 8:
            continue
        idx = sub.index.to_numpy()
        f = features[idx]
        norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
        f = f / norms

        y_bio = pd.factorize(sub["label"])[0].astype(int)
        y_ctr = pd.factorize(sub["medical_center"])[0].astype(int)

        best_k = _best_k_by_silhouette(
            f,
            k_min=max(2, k_min),
            k_max=k_max,
            random_state=random_state + pidx,
        )

        rep_scores = []
        for rep in range(repeats):
            km = KMeans(
                n_clusters=best_k,
                n_init=5,
                random_state=random_state + rep + pidx * 1000,
            )
            pred = km.fit_predict(f)
            rep_scores.append(clustering_score(pred, y_bio, y_ctr))

        pair_scores.append(float(np.mean(rep_scores)))

    if not pair_scores:
        raise RuntimeError(f"{dataset_name}: clustering failed on all 2x2 pairs")

    arr = np.array(pair_scores, dtype=float)
    return ClusteringResult(
        dataset=dataset_name,
        model_name=model_name,
        score=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        n_pairs=int(len(arr)),
    )
