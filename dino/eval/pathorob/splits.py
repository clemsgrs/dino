from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12



def cramers_v_from_counts(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=float)
    if arr.ndim != 2:
        raise ValueError("counts must be 2D")
    n = arr.sum()
    if n <= 0:
        return 0.0

    row = arr.sum(axis=1, keepdims=True)
    col = arr.sum(axis=0, keepdims=True)
    expected = row @ col / n
    mask = expected > 0
    chi2 = ((arr - expected) ** 2)[mask] / expected[mask]
    chi2 = float(chi2.sum())

    r, k = arr.shape
    denom = n * max(min(r - 1, k - 1), 1)
    return float(np.sqrt(max(chi2 / max(denom, EPS), 0.0)))



def _matrix_from_df(df: pd.DataFrame, labels: List[str], centers: List[str]) -> np.ndarray:
    mat = np.zeros((len(labels), len(centers)), dtype=int)
    for i, lbl in enumerate(labels):
        for j, ctr in enumerate(centers):
            mat[i, j] = int(((df["label"] == lbl) & (df["medical_center"] == ctr)).sum())
    return mat



def _build_max_assoc_matrix(
    row_totals: np.ndarray,
    col_totals: np.ndarray,
    preferred_cols: np.ndarray,
    capacities: np.ndarray,
) -> np.ndarray:
    """Greedy high-association matrix with fixed margins under capacities."""
    rows = row_totals.copy().astype(int)
    cols = col_totals.copy().astype(int)
    cap = capacities.copy().astype(int)
    out = np.zeros_like(cap, dtype=int)

    n_rows, n_cols = cap.shape

    for i in range(n_rows):
        # put preferred column first, then others
        pref = int(preferred_cols[i])
        order = [pref] + [j for j in range(n_cols) if j != pref]
        for j in order:
            if rows[i] <= 0:
                break
            take = min(rows[i], cols[j], cap[i, j])
            if take > 0:
                out[i, j] += take
                rows[i] -= take
                cols[j] -= take
                cap[i, j] -= take

    # fill remaining with any feasible cells
    changed = True
    while changed and rows.sum() > 0:
        changed = False
        for i in range(n_rows):
            if rows[i] <= 0:
                continue
            for j in range(n_cols):
                if cols[j] <= 0:
                    continue
                if cap[i, j] <= 0:
                    continue
                take = min(rows[i], cols[j], cap[i, j])
                if take > 0:
                    out[i, j] += take
                    rows[i] -= take
                    cols[j] -= take
                    cap[i, j] -= take
                    changed = True
                    if rows[i] <= 0:
                        break

    if rows.sum() != 0 or cols.sum() != 0:
        raise RuntimeError("Could not satisfy matrix margins under capacities")

    return out



def _project_to_margins(
    target: np.ndarray,
    row_totals: np.ndarray,
    col_totals: np.ndarray,
    capacities: np.ndarray,
    max_iter: int = 100000,
) -> np.ndarray:
    """Project float matrix to non-negative integer matrix with fixed margins and capacities."""
    n_rows, n_cols = target.shape
    mat = np.floor(target).astype(int)
    mat = np.minimum(mat, capacities)

    # quick helper
    def row_diff() -> np.ndarray:
        return row_totals - mat.sum(axis=1)

    def col_diff() -> np.ndarray:
        return col_totals - mat.sum(axis=0)

    rdiff = row_diff()
    cdiff = col_diff()

    it = 0
    while ((rdiff != 0).any() or (cdiff != 0).any()) and it < max_iter:
        it += 1

        # add where both diffs positive
        added = False
        pos_rows = np.where(rdiff > 0)[0]
        pos_cols = np.where(cdiff > 0)[0]
        if len(pos_rows) and len(pos_cols):
            for i in pos_rows:
                if rdiff[i] <= 0:
                    continue
                # prefer highest fractional residual under capacity
                candidates = []
                for j in pos_cols:
                    if cdiff[j] <= 0:
                        continue
                    if mat[i, j] >= capacities[i, j]:
                        continue
                    score = float(target[i, j] - mat[i, j])
                    candidates.append((score, j))
                if not candidates:
                    continue
                candidates.sort(reverse=True)
                j = candidates[0][1]
                mat[i, j] += 1
                added = True
                rdiff[i] -= 1
                cdiff[j] -= 1

        # remove where both diffs negative
        removed = False
        neg_rows = np.where(rdiff < 0)[0]
        neg_cols = np.where(cdiff < 0)[0]
        if len(neg_rows) and len(neg_cols):
            for i in neg_rows:
                if rdiff[i] >= 0:
                    continue
                candidates = []
                for j in neg_cols:
                    if cdiff[j] >= 0:
                        continue
                    if mat[i, j] <= 0:
                        continue
                    score = float(mat[i, j] - target[i, j])
                    candidates.append((score, j))
                if not candidates:
                    continue
                candidates.sort(reverse=True)
                j = candidates[0][1]
                mat[i, j] -= 1
                removed = True
                rdiff[i] += 1
                cdiff[j] += 1

        if not added and not removed:
            # try direct transfer along columns
            moved = False
            for i in range(n_rows):
                if rdiff[i] == 0:
                    continue
                if rdiff[i] > 0:
                    # need add in row i: borrow from another row in same column
                    for j in range(n_cols):
                        if cdiff[j] != 0:
                            continue
                        if mat[i, j] >= capacities[i, j]:
                            continue
                        donors = np.where(rdiff < 0)[0]
                        for d in donors:
                            if mat[d, j] <= 0:
                                continue
                            mat[d, j] -= 1
                            mat[i, j] += 1
                            rdiff[d] += 1
                            rdiff[i] -= 1
                            moved = True
                            break
                        if moved:
                            break
                else:
                    # need remove in row i
                    for j in range(n_cols):
                        if cdiff[j] != 0:
                            continue
                        if mat[i, j] <= 0:
                            continue
                        receivers = np.where(rdiff > 0)[0]
                        for d in receivers:
                            if mat[d, j] >= capacities[d, j]:
                                continue
                            mat[i, j] -= 1
                            mat[d, j] += 1
                            rdiff[i] += 1
                            rdiff[d] -= 1
                            moved = True
                            break
                        if moved:
                            break
                if moved:
                    break

            if not moved:
                break

    if (row_totals - mat.sum(axis=1)).any() or (col_totals - mat.sum(axis=0)).any():
        raise RuntimeError("Projection to margins failed")

    return mat



def _choose_train_id_slides(
    id_df: pd.DataFrame,
    labels: List[str],
    centers: List[str],
    id_test_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    train_parts = []
    id_test_parts = []

    for lbl in labels:
        for ctr in centers:
            cell = id_df[(id_df["label"] == lbl) & (id_df["medical_center"] == ctr)]
            slides = sorted(cell["slide_id"].unique().tolist())
            if not slides:
                continue
            n_test = max(1, int(round(len(slides) * id_test_fraction))) if len(slides) > 1 else 0
            if n_test >= len(slides):
                n_test = len(slides) - 1
            if n_test > 0:
                test_slides = set(rng.choice(slides, size=n_test, replace=False).tolist())
            else:
                test_slides = set()
            train_slides = set(slides) - test_slides
            train_parts.append(cell[cell["slide_id"].isin(train_slides)])
            id_test_parts.append(cell[cell["slide_id"].isin(test_slides)])

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True) if train_parts else id_df.iloc[0:0].copy()
    id_test_df = pd.concat(id_test_parts, axis=0).reset_index(drop=True) if id_test_parts else id_df.iloc[0:0].copy()

    # leakage guard
    train_slides = set(train_df["slide_id"].unique().tolist())
    id_slides = set(id_test_df["slide_id"].unique().tolist())
    if train_slides & id_slides:
        raise RuntimeError("Slide leakage detected between train and id_test")

    return train_df, id_test_df



def _sample_cell_rows(
    cell_df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if n <= 0:
        return cell_df.iloc[0:0].copy()
    if len(cell_df) < n:
        raise RuntimeError(f"Not enough samples in cell: need {n}, have {len(cell_df)}")
    idx = rng.choice(cell_df.index.values, size=n, replace=False)
    return cell_df.loc[idx]



def _sample_from_matrix(
    train_pool: pd.DataFrame,
    labels: List[str],
    centers: List[str],
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    parts = []
    for i, lbl in enumerate(labels):
        for j, ctr in enumerate(centers):
            n = int(matrix[i, j])
            cell = train_pool[(train_pool["label"] == lbl) & (train_pool["medical_center"] == ctr)]
            parts.append(_sample_cell_rows(cell, n, rng))
    out = pd.concat(parts, axis=0).reset_index(drop=True) if parts else train_pool.iloc[0:0].copy()
    return out



def generate_apd_splits(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    repetitions: int,
    correlation_levels: Sequence[float],
    id_centers: Sequence[str],
    ood_centers: Sequence[str],
    id_test_fraction: float,
    seed: int,
) -> List[pd.DataFrame]:
    """Generate and persist APD splits with fixed class/center margins across rho levels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(df["label"].unique().tolist())
    centers = sorted([c for c in id_centers if c in set(df["medical_center"].unique().tolist())])
    if not labels or not centers:
        raise ValueError(f"{dataset_name}: empty labels/centers for APD")

    id_df = df[df["medical_center"].isin(centers)].copy()
    ood_df = df[df["medical_center"].isin(set(ood_centers))].copy()

    if len(ood_df) == 0:
        raise ValueError(f"{dataset_name}: OOD set is empty")

    all_splits: List[pd.DataFrame] = []

    for rep in range(repetitions):
        rep_rng = np.random.default_rng(seed + rep)

        train_pool, id_test_df = _choose_train_id_slides(
            id_df=id_df,
            labels=labels,
            centers=centers,
            id_test_fraction=id_test_fraction,
            random_state=seed + rep,
        )

        # fixed OOD test for this repetition: sample by slide-balanced class-center minimum
        ood_parts = []
        for lbl in labels:
            for ctr in sorted(ood_df["medical_center"].unique().tolist()):
                cell = ood_df[(ood_df["label"] == lbl) & (ood_df["medical_center"] == ctr)]
                if len(cell) == 0:
                    continue
                ood_parts.append(cell)
        rep_ood_df = pd.concat(ood_parts, axis=0).reset_index(drop=True) if ood_parts else ood_df

        avail = _matrix_from_df(train_pool, labels, centers)
        if (avail <= 0).any():
            raise RuntimeError(
                f"{dataset_name}: some train pool class-center cells are empty; cannot build parity APD splits"
            )

        # Use half of minimum cell capacity to leave room for biased redistribution.
        # Without this, if all cells have equal capacity, max_assoc == uniform and
        # we can't create biased splits.
        base = int(avail.min()) // 2
        if base <= 0:
            raise RuntimeError(f"{dataset_name}: base cell size is zero (need at least 2 samples per cell)")

        row_totals = np.full(len(labels), base * len(centers), dtype=int)
        col_totals = np.full(len(centers), base * len(labels), dtype=int)
        uniform = np.full((len(labels), len(centers)), base, dtype=int)

        preferred_cols = np.array([i % len(centers) for i in range(len(labels))], dtype=int)
        max_assoc = _build_max_assoc_matrix(
            row_totals=row_totals,
            col_totals=col_totals,
            preferred_cols=preferred_cols,
            capacities=avail,
        )

        for split_idx, rho in enumerate(correlation_levels):
            rho = float(rho)
            rho = min(max(rho, 0.0), 1.0)
            target = (1.0 - rho) * uniform.astype(float) + rho * max_assoc.astype(float)
            split_matrix = _project_to_margins(
                target=target,
                row_totals=row_totals,
                col_totals=col_totals,
                capacities=avail,
            )

            train_df = _sample_from_matrix(train_pool, labels, centers, split_matrix, rep_rng)

            # add split fields
            train_part = train_df.copy()
            train_part["partition"] = "train"
            id_part = id_test_df.copy()
            id_part["partition"] = "id_test"
            ood_part = rep_ood_df.copy()
            ood_part["partition"] = "ood_test"

            merged = pd.concat([train_part, id_part, ood_part], axis=0).reset_index(drop=True)
            merged["rep"] = int(rep)
            merged["split_id"] = int(split_idx + 1)
            merged["correlation_level"] = float(rho)
            merged["cramers_v_target"] = float(rho)
            merged["dataset"] = dataset_name

            # Compute realized Cramer's V on training split
            ctab = pd.crosstab(merged[merged["partition"] == "train"]["label"], merged[merged["partition"] == "train"]["medical_center"])
            merged["cramers_v_realized"] = float(cramers_v_from_counts(ctab.values))

            rep_dir = output_dir / dataset_name / f"rep_{rep:02d}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            out_csv = rep_dir / f"split_{split_idx+1:02d}.csv"
            merged.to_csv(out_csv, index=False)

            all_splits.append(merged)

    return all_splits
