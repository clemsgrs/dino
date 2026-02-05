from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dino.eval.pathorob.allocations import (
    get_paper_allocations,
    get_paper_v_levels,
    scale_allocation,
)

logger = logging.getLogger("dino")

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
    mode: Literal["interpolate", "paper"] = "paper",
) -> List[pd.DataFrame]:
    """Generate and persist APD splits with fixed class/center margins across rho levels.

    Args:
        df: DataFrame with columns [sample_id, image_path, label, medical_center, slide_id]
        output_dir: Directory to save split CSVs
        dataset_name: Name of dataset (used for paper mode lookup and output paths)
        repetitions: Number of random repetitions
        correlation_levels: List of target Cramér's V values (ignored if mode="paper")
        id_centers: In-distribution medical centers for training
        ood_centers: Out-of-distribution medical centers for OOD test
        id_test_fraction: Fraction of ID slides to hold out for ID test
        seed: Random seed for reproducibility
        mode: Split generation mode:
            - "paper": Use exact allocation matrices from the PathoROB paper (default)
            - "interpolate": Interpolate between uniform and max-association matrices

    Returns:
        List of DataFrames, one per (rep, correlation_level) combination
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(df["label"].unique().tolist())
    centers = sorted([c for c in id_centers if c in set(df["medical_center"].unique().tolist())])
    if not labels or not centers:
        raise ValueError(f"{dataset_name}: empty labels/centers for APD")

    id_df = df[df["medical_center"].isin(centers)].copy()
    ood_df = df[df["medical_center"].isin(set(ood_centers))].copy()

    if len(ood_df) == 0:
        raise ValueError(f"{dataset_name}: OOD set is empty")

    # Paper mode: use exact paper allocations and V levels
    paper_allocations = None
    if mode == "paper":
        try:
            paper_allocations = get_paper_allocations(dataset_name)
            paper_v_levels = get_paper_v_levels(dataset_name)
            # Override correlation_levels with paper values
            correlation_levels = paper_v_levels
            logger.info(f"[APD] Using paper allocations for {dataset_name} with V levels: {paper_v_levels}")
        except ValueError as e:
            logger.warning(f"[APD] Paper allocations not available for {dataset_name}: {e}. Falling back to interpolate mode.")
            mode = "interpolate"

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

        # Compute base allocations depending on mode
        if mode == "paper" and paper_allocations is not None:
            # Scale paper allocations to available data
            # Use minimum available cell to determine scale factor
            paper_base = paper_allocations[0.0]  # Use V=0 as reference
            paper_total = int(paper_base.sum())

            # Find maximum training size we can use (limited by smallest cell ratio)
            scale_factors = avail / np.maximum(paper_base, 1)
            scale_factors[paper_base == 0] = np.inf  # Don't limit by zero cells
            max_scale = float(scale_factors.min())

            # Target training size: scale paper total, but cap at available
            target_train_total = min(int(paper_total * max_scale), int(avail.sum()))

            # For V=1.0, some cells are zero - ensure we have enough in diagonal cells
            v1_alloc = paper_allocations[1.0]
            for i in range(v1_alloc.shape[0]):
                for j in range(v1_alloc.shape[1]):
                    if v1_alloc[i, j] > 0:
                        # This cell needs capacity for V=1
                        needed_ratio = v1_alloc[i, j] / paper_total
                        needed = int(needed_ratio * target_train_total)
                        if avail[i, j] < needed:
                            # Reduce target to fit
                            target_train_total = int(avail[i, j] / needed_ratio)

            logger.info(f"[APD] Paper mode: scaling from {paper_total} to {target_train_total} training samples")
        else:
            # Interpolate mode: use half of minimum cell capacity
            base = int(avail.min()) // 2
            if base <= 0:
                raise RuntimeError(f"{dataset_name}: base cell size is zero (need at least 2 samples per cell)")
            target_train_total = base * len(labels) * len(centers)

        # Precompute allocation matrices for each correlation level
        allocation_matrices = {}
        for rho in correlation_levels:
            rho = float(rho)
            rho = min(max(rho, 0.0), 1.0)

            if mode == "paper" and paper_allocations is not None:
                # Scale paper allocation to target size
                if rho in paper_allocations:
                    paper_alloc = paper_allocations[rho]
                else:
                    # Interpolate between nearest paper levels
                    v_levels = sorted(paper_allocations.keys())
                    v_lo = max(v for v in v_levels if v <= rho)
                    v_hi = min(v for v in v_levels if v >= rho)
                    if v_lo == v_hi:
                        paper_alloc = paper_allocations[v_lo]
                    else:
                        t = (rho - v_lo) / (v_hi - v_lo)
                        paper_alloc = ((1 - t) * paper_allocations[v_lo] + t * paper_allocations[v_hi])

                split_matrix = scale_allocation(paper_alloc, target_train_total)

                # Verify we have capacity
                if (split_matrix > avail).any():
                    # Reduce cells that exceed capacity
                    excess = np.maximum(split_matrix - avail, 0)
                    split_matrix = split_matrix - excess
                    # Redistribute excess to other cells if possible
                    deficit = int(excess.sum())
                    if deficit > 0:
                        slack = avail - split_matrix
                        for _ in range(deficit):
                            if slack.max() <= 0:
                                break
                            idx = np.unravel_index(np.argmax(slack), slack.shape)
                            split_matrix[idx] += 1
                            slack[idx] -= 1
            else:
                # Interpolate mode
                base = target_train_total // (len(labels) * len(centers))
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

                target = (1.0 - rho) * uniform.astype(float) + rho * max_assoc.astype(float)
                split_matrix = _project_to_margins(
                    target=target,
                    row_totals=row_totals,
                    col_totals=col_totals,
                    capacities=avail,
                )

            allocation_matrices[rho] = split_matrix

        for split_idx, rho in enumerate(correlation_levels):
            rho = float(rho)
            split_matrix = allocation_matrices[rho]

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
