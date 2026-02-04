from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "image_path",
    "label",
    "medical_center",
    "slide_id",
)


@dataclass
class PairSpec:
    dataset: str
    pair_id: str
    classes: Tuple[str, str]
    centers: Tuple[str, str]



def _normalize_str(v: object) -> str:
    return str(v).strip()



def ensure_required_columns(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")



def load_manifest(csv_path: str, dataset_name: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found for {dataset_name}: {csv_path}")

    df = pd.read_csv(path)
    ensure_required_columns(df, f"manifest {csv_path}")

    out = df.copy()
    if "sample_id" not in out.columns:
        out["sample_id"] = [f"{dataset_name}_{i:09d}" for i in range(len(out))]

    out["dataset"] = dataset_name
    out["label"] = out["label"].map(_normalize_str)
    out["medical_center"] = out["medical_center"].map(_normalize_str)
    out["slide_id"] = out["slide_id"].map(_normalize_str)
    out["image_path"] = out["image_path"].map(_normalize_str)

    # keep only existing images
    exists_mask = out["image_path"].map(lambda p: Path(p).exists())
    if not bool(exists_mask.all()):
        out = out.loc[exists_mask].reset_index(drop=True)

    return out



def infer_2x2_pairs(
    df: pd.DataFrame,
    dataset_name: str,
    max_pairs: Optional[int] = None,
    random_state: int = 0,
) -> List[PairSpec]:
    """Build all valid (2 classes x 2 centers) pair specs from a manifest.

    A pair is valid if all four class-center cells have >=1 sample.
    """
    labels = sorted(df["label"].unique().tolist())
    centers = sorted(df["medical_center"].unique().tolist())

    pairs: List[PairSpec] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            c1, c2 = labels[i], labels[j]
            sub = df[df["label"].isin([c1, c2])]
            sub_centers = sorted(sub["medical_center"].unique().tolist())
            for a in range(len(sub_centers)):
                for b in range(a + 1, len(sub_centers)):
                    m1, m2 = sub_centers[a], sub_centers[b]
                    ok = True
                    for lbl in (c1, c2):
                        for ctr in (m1, m2):
                            n = int(
                                ((sub["label"] == lbl) & (sub["medical_center"] == ctr)).sum()
                            )
                            if n <= 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        pair_id = f"{dataset_name}::{c1}__{c2}::{m1}__{m2}"
                        pairs.append(
                            PairSpec(
                                dataset=dataset_name,
                                pair_id=pair_id,
                                classes=(c1, c2),
                                centers=(m1, m2),
                            )
                        )

    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(random_state)
        idxs = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in sorted(idxs.tolist())]

    return pairs



def subset_by_pair(df: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    return df[
        df["label"].isin(pair.classes) & df["medical_center"].isin(pair.centers)
    ].copy()



def align_centers(df: pd.DataFrame, include_centers: Sequence[str]) -> pd.DataFrame:
    include = set(include_centers)
    return df[df["medical_center"].isin(include)].copy()



def choose_case_subset(
    df: pd.DataFrame,
    per_cell_cases: int,
    random_state: int,
) -> pd.DataFrame:
    """Choose up to N cases (slides) per class-center cell.

    This mirrors the paper's case subsampling behavior for Tolkach-like settings.
    """
    rng = np.random.default_rng(random_state)
    chunks = []
    for (label, center), cell in df.groupby(["label", "medical_center"], sort=True):
        cases = sorted(cell["slide_id"].unique().tolist())
        if len(cases) <= per_cell_cases:
            chosen = set(cases)
        else:
            chosen = set(rng.choice(cases, size=per_cell_cases, replace=False).tolist())
        chunks.append(cell[cell["slide_id"].isin(chosen)])

    if not chunks:
        return df.iloc[0:0].copy()
    return pd.concat(chunks, axis=0).reset_index(drop=True)
