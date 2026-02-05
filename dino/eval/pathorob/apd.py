from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logger = logging.getLogger("dino")


@dataclass
class APDResult:
    dataset: str
    model_name: str
    apd_id: float
    apd_ood: float
    apd_avg: float
    apd_id_std: float
    apd_ood_std: float
    apd_avg_std: float
    # Per-rho accuracies: {rho_value: (mean_acc, std_acc)}
    acc_id_by_rho: Dict[float, Tuple[float, float]]
    acc_ood_by_rho: Dict[float, Tuple[float, float]]



def _train_linear_probe(train_x: np.ndarray, train_y: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(train_x, train_y)
    return clf



def _apd_from_split_acc(acc_by_split: Dict[int, float]) -> float:
    if 1 not in acc_by_split:
        raise RuntimeError("Split 1 (balanced) missing for APD")
    base = float(acc_by_split[1])
    keys = sorted([k for k in acc_by_split.keys() if k != 1])
    if not keys:
        return 0.0
    drops = [((float(acc_by_split[k]) - base) / max(base, 1e-12)) for k in keys]
    return float(np.mean(drops))



def compute_apd(
    dataset_name: str,
    model_name: str,
    features: np.ndarray,
    all_splits: Sequence[pd.DataFrame],
    seed: int,
    verbose: bool = False,
) -> APDResult:
    """Compute APD over generated split manifests for one dataset+model."""
    if len(all_splits) == 0:
        raise RuntimeError(f"{dataset_name}: no APD splits provided")

    # Track accuracies by (rep, split_id) for APD computation
    id_by_rep: Dict[int, Dict[int, float]] = {}
    ood_by_rep: Dict[int, Dict[int, float]] = {}
    # Track accuracies by (rep, rho) for per-rho logging
    id_by_rho: Dict[float, List[float]] = {}
    ood_by_rho: Dict[float, List[float]] = {}

    for i, split_df in enumerate(all_splits):
        rep = int(split_df["rep"].iloc[0])
        split_id = int(split_df["split_id"].iloc[0])
        rho = float(split_df["correlation_level"].iloc[0])

        # merge back to feature rows via sample_id
        cols = ["sample_id", "partition", "label"]
        if not set(cols).issubset(split_df.columns):
            raise RuntimeError("Split dataframe must include sample_id, partition, label")

        # We rely on a per-split feature alignment table created by caller:
        # split_df['feature_index'].
        if "feature_index" in split_df.columns:
            feat_idx = split_df["feature_index"].to_numpy(dtype=int)
        else:
            raise RuntimeError("Split dataframe missing feature_index")

        y = pd.factorize(split_df["label"])[0].astype(int)

        train_mask = split_df["partition"] == "train"
        id_mask = split_df["partition"] == "id_test"
        ood_mask = split_df["partition"] == "ood_test"

        if train_mask.sum() == 0 or id_mask.sum() == 0 or ood_mask.sum() == 0:
            raise RuntimeError(f"{dataset_name}: split {split_id} rep {rep} missing partitions")

        x_train = features[feat_idx[train_mask.to_numpy()]]
        y_train = y[train_mask.to_numpy()]
        x_id = features[feat_idx[id_mask.to_numpy()]]
        y_id = y[id_mask.to_numpy()]
        x_ood = features[feat_idx[ood_mask.to_numpy()]]
        y_ood = y[ood_mask.to_numpy()]

        # Debug: log training set distribution per rho
        train_labels = split_df[split_df["partition"] == "train"]["label"].value_counts().to_dict()
        train_centers = split_df[split_df["partition"] == "train"]["medical_center"].value_counts().to_dict()
        logger.info(
            f"[APD] split={i}/{len(all_splits)} rep={rep} rho={rho:.2f}: "
            f"train={len(x_train)} samples, labels={train_labels}, centers={train_centers}"
        )

        clf = _train_linear_probe(x_train, y_train, seed + rep * 100 + split_id)

        pred_id = clf.predict(x_id)
        pred_ood = clf.predict(x_ood)

        acc_id = float(accuracy_score(y_id, pred_id))
        acc_ood = float(accuracy_score(y_ood, pred_ood))

        if verbose:
            logger.info(f"[APD] split={i}/{len(all_splits)} rep={rep} rho={rho:.2f}: acc_id={acc_id:.4f}, acc_ood={acc_ood:.4f}")

        id_by_rep.setdefault(rep, {})[split_id] = acc_id
        ood_by_rep.setdefault(rep, {})[split_id] = acc_ood
        id_by_rho.setdefault(rho, []).append(acc_id)
        ood_by_rho.setdefault(rho, []).append(acc_ood)

    # Compute APD across repetitions
    apd_id = []
    apd_ood = []
    apd_avg = []

    reps = sorted(set(id_by_rep.keys()) & set(ood_by_rep.keys()))
    for rep in reps:
        rep_apd_id = _apd_from_split_acc(id_by_rep[rep])
        rep_apd_ood = _apd_from_split_acc(ood_by_rep[rep])
        apd_id.append(rep_apd_id)
        apd_ood.append(rep_apd_ood)
        apd_avg.append((rep_apd_id + rep_apd_ood) / 2.0)

    apd_id_arr = np.array(apd_id, dtype=float)
    apd_ood_arr = np.array(apd_ood, dtype=float)
    apd_avg_arr = np.array(apd_avg, dtype=float)

    # Aggregate per-rho accuracies: {rho: (mean, std)}
    acc_id_by_rho: Dict[float, Tuple[float, float]] = {}
    acc_ood_by_rho: Dict[float, Tuple[float, float]] = {}
    for rho in sorted(id_by_rho.keys()):
        arr = np.array(id_by_rho[rho], dtype=float)
        acc_id_by_rho[rho] = (float(arr.mean()), float(arr.std(ddof=0)))
    for rho in sorted(ood_by_rho.keys()):
        arr = np.array(ood_by_rho[rho], dtype=float)
        acc_ood_by_rho[rho] = (float(arr.mean()), float(arr.std(ddof=0)))

    return APDResult(
        dataset=dataset_name,
        model_name=model_name,
        apd_id=float(apd_id_arr.mean() if len(apd_id_arr) else 0.0),
        apd_ood=float(apd_ood_arr.mean() if len(apd_ood_arr) else 0.0),
        apd_avg=float(apd_avg_arr.mean() if len(apd_avg_arr) else 0.0),
        apd_id_std=float(apd_id_arr.std(ddof=0) if len(apd_id_arr) else 0.0),
        apd_ood_std=float(apd_ood_arr.std(ddof=0) if len(apd_ood_arr) else 0.0),
        apd_avg_std=float(apd_avg_arr.std(ddof=0) if len(apd_avg_arr) else 0.0),
        acc_id_by_rho=acc_id_by_rho,
        acc_ood_by_rho=acc_ood_by_rho,
    )
