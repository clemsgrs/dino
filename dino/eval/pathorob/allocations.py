"""Paper-specified allocation matrices for PathoROB benchmark datasets.

These matrices define the exact training set allocations used in the PathoROB paper
(Supplementary Note D, Figures 14-16) for each Cramér's V correlation level.

Each matrix has shape (n_classes, n_centers) where:
- Rows correspond to biological classes
- Columns correspond to medical centers
- Values are the number of training patches to sample from each cell

All matrices within a dataset have fixed row and column totals (marginals),
ensuring the same total training size across correlation levels.
"""

from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# CAMELYON (Figure 14)
# =============================================================================
# Classes: [Normal, Tumor]
# Centers: [RUMC, UMCU]
# Training total: 8400 (4200 per class, 4200 per center)

CAMELYON_CLASSES = ["normal", "tumor"]
CAMELYON_ID_CENTERS = ["RUMC", "UMCU"]
CAMELYON_OOD_CENTERS = ["CWZ", "RST", "LPON"]

CAMELYON_ALLOCATIONS: Dict[float, np.ndarray] = {
    0.00: np.array([[2100, 2100], [2100, 2100]]),
    0.14: np.array([[1800, 2400], [2400, 1800]]),
    0.29: np.array([[1500, 2700], [2700, 1500]]),
    0.43: np.array([[1200, 3000], [3000, 1200]]),
    0.57: np.array([[900, 3300], [3300, 900]]),
    0.71: np.array([[600, 3600], [3600, 600]]),
    0.86: np.array([[300, 3900], [3900, 300]]),
    1.00: np.array([[0, 4200], [4200, 0]]),
}

CAMELYON_V_LEVELS = [0.00, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.00]


# =============================================================================
# Helper functions
# =============================================================================

def get_paper_allocations(dataset: str) -> Dict[float, np.ndarray]:
    """Get paper allocation matrices for a dataset.

    Args:
        dataset: Dataset name ("camelyon", "tcga_4x4", "tolkach_esca")

    Returns:
        Dictionary mapping V level to allocation matrix
    """
    dataset = dataset.lower()
    if dataset in ("camelyon", "camelyon16", "camelyon17"):
        return CAMELYON_ALLOCATIONS
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available: camelyon")


def get_paper_v_levels(dataset: str) -> List[float]:
    """Get paper V levels for a dataset.

    Args:
        dataset: Dataset name

    Returns:
        List of Cramér's V levels used in the paper
    """
    dataset = dataset.lower()
    if dataset in ("camelyon", "camelyon16", "camelyon17"):
        return CAMELYON_V_LEVELS.copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available: camelyon")


def get_paper_metadata(dataset: str) -> Tuple[List[str], List[str], List[str]]:
    """Get paper class and center labels for a dataset.

    Args:
        dataset: Dataset name

    Returns:
        Tuple of (classes, id_centers, ood_centers)
    """
    dataset = dataset.lower()
    if dataset in ("camelyon", "camelyon16", "camelyon17"):
        return CAMELYON_CLASSES.copy(), CAMELYON_ID_CENTERS.copy(), CAMELYON_OOD_CENTERS.copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available: camelyon")


def scale_allocation(
    alloc: np.ndarray,
    target_total: int,
    min_cell: int = 0,
) -> np.ndarray:
    """Scale allocation matrix to target total while preserving proportions.

    Args:
        alloc: Original allocation matrix
        target_total: Desired total count
        min_cell: Minimum count per non-zero cell (0 to allow zeros)

    Returns:
        Scaled allocation matrix with same shape
    """
    current_total = alloc.sum()
    if current_total == 0:
        return alloc.copy()

    # Scale proportionally
    scale = target_total / current_total
    scaled = alloc.astype(float) * scale

    # Round to integers while preserving totals
    result = np.floor(scaled).astype(int)

    # Distribute remaining counts to cells with largest fractional parts
    remainder = target_total - result.sum()
    if remainder > 0:
        fractions = scaled - result
        # Don't add to cells that should be zero
        fractions[alloc == 0] = -1
        flat_idx = np.argsort(fractions.ravel())[::-1]
        for i in range(int(remainder)):
            idx = np.unravel_index(flat_idx[i], result.shape)
            result[idx] += 1
    elif remainder < 0:
        # Need to remove some (rare due to floor)
        for _ in range(int(-remainder)):
            # Remove from largest cell
            idx = np.unravel_index(np.argmax(result), result.shape)
            result[idx] -= 1

    # Enforce minimum cell counts
    if min_cell > 0:
        mask = (alloc > 0) & (result < min_cell)
        deficit = (min_cell - result[mask]).sum()
        if deficit > 0:
            # Borrow from largest cells
            result[mask] = min_cell
            excess = result.sum() - target_total
            while excess > 0:
                # Find largest cell not at minimum
                candidates = result.copy()
                candidates[result <= min_cell] = 0
                if candidates.max() <= min_cell:
                    break
                idx = np.unravel_index(np.argmax(candidates), result.shape)
                take = min(excess, result[idx] - min_cell)
                result[idx] -= take
                excess -= take

    return result


def interpolate_allocation(
    v_target: float,
    allocations: Dict[float, np.ndarray],
) -> np.ndarray:
    """Interpolate between paper allocations for intermediate V values.

    Args:
        v_target: Target Cramér's V value
        allocations: Dictionary of V -> allocation matrix

    Returns:
        Interpolated allocation matrix (integer)
    """
    v_levels = sorted(allocations.keys())

    # Exact match
    if v_target in allocations:
        return allocations[v_target].copy()

    # Clamp to range
    if v_target <= v_levels[0]:
        return allocations[v_levels[0]].copy()
    if v_target >= v_levels[-1]:
        return allocations[v_levels[-1]].copy()

    # Find bracketing levels
    v_lo = max(v for v in v_levels if v < v_target)
    v_hi = min(v for v in v_levels if v > v_target)

    # Linear interpolation
    t = (v_target - v_lo) / (v_hi - v_lo)
    alloc_lo = allocations[v_lo].astype(float)
    alloc_hi = allocations[v_hi].astype(float)
    interpolated = (1 - t) * alloc_lo + t * alloc_hi

    # Round to integers preserving totals
    result = np.round(interpolated).astype(int)

    # Adjust to match original totals
    target_row = allocations[v_lo].sum(axis=1)
    target_col = allocations[v_lo].sum(axis=0)

    # Simple adjustment: fix row totals first, then columns
    for i in range(result.shape[0]):
        diff = target_row[i] - result[i].sum()
        if diff != 0:
            j = np.argmax(result[i]) if diff < 0 else np.argmin(result[i])
            result[i, j] += diff

    for j in range(result.shape[1]):
        diff = target_col[j] - result[:, j].sum()
        if diff != 0:
            i = np.argmax(result[:, j]) if diff < 0 else np.argmin(result[:, j])
            result[i, j] += diff

    return result
