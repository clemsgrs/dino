from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

from .datasets import PairSpec, infer_2x2_pairs, subset_by_pair


@dataclass
class RIResult:
    dataset: str
    model_name: str
    k: int
    value: float
    std: float
    n_pairs: int



def _normalized_ri_from_neighbors(
    labels: np.ndarray,
    centers: np.ndarray,
    neigh_idx: np.ndarray,
    k: int,
) -> float:
    neigh = neigh_idx[:, :k]
    sample_labels = labels[:, None]
    sample_centers = centers[:, None]

    neigh_labels = labels[neigh]
    neigh_centers = centers[neigh]

    so = np.logical_and(neigh_labels == sample_labels, neigh_centers != sample_centers).sum()
    os = np.logical_and(neigh_labels != sample_labels, neigh_centers == sample_centers).sum()

    denom = float(so + os)
    if denom <= 0:
        return 0.5
    return float(so / denom)



def _prepare_neighbors(
    features: np.ndarray,
    slide_ids: np.ndarray,
    kmax: int,
) -> np.ndarray:
    # +1 because we will drop self neighbor
    nn = NearestNeighbors(n_neighbors=min(kmax + 64, len(features) - 1), metric="cosine")
    nn.fit(features)
    _, neigh = nn.kneighbors(features)

    # remove same-slide and self; keep first kmax valid neighbors
    out = np.full((len(features), kmax), -1, dtype=int)
    for i in range(len(features)):
        vals = []
        for j in neigh[i].tolist():
            if j == i:
                continue
            if slide_ids[j] == slide_ids[i]:
                continue
            vals.append(j)
            if len(vals) == kmax:
                break
        if len(vals) < kmax:
            # fallback: allow same slide to avoid hard failure on tiny sets
            for j in neigh[i].tolist():
                if j == i:
                    continue
                if j in vals:
                    continue
                vals.append(j)
                if len(vals) == kmax:
                    break
        if len(vals) < kmax:
            raise RuntimeError("Not enough neighbors to compute RI")
        out[i, :] = np.array(vals, dtype=int)
    return out



def _optimal_k_by_knn_balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    slide_ids: np.ndarray,
    k_values: Sequence[int],
) -> int:
    kmax = int(max(k_values))
    neigh = _prepare_neighbors(features, slide_ids, kmax)

    best_k = int(k_values[0])
    best_score = -1.0

    for k in k_values:
        topk = neigh[:, : int(k)]
        neigh_labels = labels[topk]
        # majority vote
        pred = []
        for row in neigh_labels:
            vals, cnt = np.unique(row, return_counts=True)
            pred.append(int(vals[np.argmax(cnt)]))
        score = balanced_accuracy_score(labels, np.array(pred, dtype=int))
        if score > best_score:
            best_score = float(score)
            best_k = int(k)

    return best_k



def compute_ri(
    dataset_name: str,
    model_name: str,
    features: np.ndarray,
    manifest_df: pd.DataFrame,
    policy: str,
    fixed_k: int,
    k_candidates: Sequence[int],
    max_pairs: Optional[int],
    random_state: int,
) -> RIResult:
    """Compute PathoROB-style RI on all valid 2x2 pairs and average."""
    df = manifest_df.reset_index(drop=True).copy()

    # infer valid 2x2 subsets
    pairs = infer_2x2_pairs(df, dataset_name=dataset_name, max_pairs=max_pairs, random_state=random_state)
    if not pairs:
        raise RuntimeError(f"{dataset_name}: no valid 2x2 pairs for RI")

    if policy == "paper_median_fixed":
        k = int(fixed_k)
    elif policy == "knn_bacc":
        k = _optimal_k_by_knn_balanced_accuracy(
            features=features,
            labels=pd.factorize(df["label"])[0].astype(int),
            slide_ids=df["slide_id"].astype(str).values,
            k_values=k_candidates,
        )
    else:
        raise ValueError(f"Unsupported RI k-selection policy: {policy}")

    values: List[float] = []
    for pair in pairs:
        sub = subset_by_pair(df, pair)
        if len(sub) <= k + 1:
            continue
        idx = sub.index.to_numpy()
        f = features[idx]

        # Normalize embeddings for cosine parity
        norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
        f = f / norms

        lbl = pd.factorize(sub["label"])[0].astype(int)
        ctr = pd.factorize(sub["medical_center"])[0].astype(int)
        sid = sub["slide_id"].astype(str).values

        neigh = _prepare_neighbors(f, sid, k)
        ri = _normalized_ri_from_neighbors(lbl, ctr, neigh, k)
        values.append(float(ri))

    if not values:
        raise RuntimeError(f"{dataset_name}: RI failed on all inferred 2x2 pairs")

    arr = np.array(values, dtype=float)
    return RIResult(
        dataset=dataset_name,
        model_name=model_name,
        k=int(k),
        value=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        n_pairs=int(len(arr)),
    )
