"""Tests for CAMELYON split generation following PathoROB paper specification.

These tests verify that:
1. Cramér's V calculation is correct
2. Paper-specified allocations produce expected V values
3. Split generation maintains required invariants (no leakage, determinism, etc.)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dino.eval.pathorob.splits import cramers_v_from_counts, generate_apd_splits


# =============================================================================
# Paper-specified allocations for CAMELYON (Figure 14, Supplementary Note D)
# =============================================================================
# Matrix layout: rows = [Normal, Tumor], cols = [RUMC, UMCU]
# Each split has same row/column totals: 4200 per class, 4200 per center

CAMELYON_PAPER_ALLOCATIONS = {
    0.00: np.array([[2100, 2100], [2100, 2100]]),
    0.14: np.array([[1800, 2400], [2400, 1800]]),
    0.29: np.array([[1500, 2700], [2700, 1500]]),
    0.43: np.array([[1200, 3000], [3000, 1200]]),
    0.57: np.array([[900, 3300], [3300, 900]]),
    0.71: np.array([[600, 3600], [3600, 600]]),
    0.86: np.array([[300, 3900], [3900, 300]]),
    1.00: np.array([[0, 4200], [4200, 0]]),
}

CAMELYON_PAPER_V_LEVELS = [0.00, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.00]


class TestCramersVCalculation(unittest.TestCase):
    """Test Cramér's V formula correctness."""

    def test_perfect_association_v_equals_one(self):
        """Diagonal matrix (perfect association) should have V=1."""
        counts = np.array([[100, 0], [0, 100]])
        v = cramers_v_from_counts(counts)
        self.assertAlmostEqual(v, 1.0, places=5)

    def test_no_association_v_equals_zero(self):
        """Uniform matrix (no association) should have V=0."""
        counts = np.array([[50, 50], [50, 50]])
        v = cramers_v_from_counts(counts)
        self.assertAlmostEqual(v, 0.0, places=5)

    def test_partial_association(self):
        """Intermediate case should give V between 0 and 1."""
        counts = np.array([[70, 30], [30, 70]])
        v = cramers_v_from_counts(counts)
        self.assertGreater(v, 0.0)
        self.assertLess(v, 1.0)

    def test_asymmetric_totals(self):
        """V should still be valid with unequal row/column totals."""
        counts = np.array([[80, 20], [40, 60]])
        v = cramers_v_from_counts(counts)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_single_cell_nonzero(self):
        """Edge case: only one cell has counts."""
        counts = np.array([[100, 0], [0, 0]])
        v = cramers_v_from_counts(counts)
        # Degenerate case, but should not crash
        self.assertGreaterEqual(v, 0.0)

    def test_empty_matrix(self):
        """Empty matrix should return 0."""
        counts = np.array([[0, 0], [0, 0]])
        v = cramers_v_from_counts(counts)
        self.assertEqual(v, 0.0)


class TestPaperAllocationsVValues(unittest.TestCase):
    """Verify paper allocation matrices produce expected Cramér's V values."""

    def test_paper_allocations_match_target_v(self):
        """Each paper allocation should produce V close to its target."""
        for v_target, alloc in CAMELYON_PAPER_ALLOCATIONS.items():
            with self.subTest(v_target=v_target):
                v_realized = cramers_v_from_counts(alloc)
                # Allow small tolerance due to discrete nature
                self.assertAlmostEqual(
                    v_realized, v_target, delta=0.02,
                    msg=f"V={v_target}: expected {v_target}, got {v_realized}"
                )

    def test_paper_allocations_have_fixed_margins(self):
        """All paper allocations should have same row/column totals."""
        expected_row_totals = np.array([4200, 4200])
        expected_col_totals = np.array([4200, 4200])

        for v_target, alloc in CAMELYON_PAPER_ALLOCATIONS.items():
            with self.subTest(v_target=v_target):
                row_totals = alloc.sum(axis=1)
                col_totals = alloc.sum(axis=0)
                np.testing.assert_array_equal(row_totals, expected_row_totals)
                np.testing.assert_array_equal(col_totals, expected_col_totals)

    def test_paper_v_levels_are_ordered(self):
        """V levels should be monotonically increasing."""
        for i in range(len(CAMELYON_PAPER_V_LEVELS) - 1):
            self.assertLess(
                CAMELYON_PAPER_V_LEVELS[i],
                CAMELYON_PAPER_V_LEVELS[i + 1]
            )


class TestAllocationScaling(unittest.TestCase):
    """Test that allocation matrices can be scaled while preserving V."""

    def _scale_allocation(self, alloc: np.ndarray, target_total: int) -> np.ndarray:
        """Scale allocation matrix to target total, preserving proportions."""
        current_total = alloc.sum()
        if current_total == 0:
            return alloc.copy()
        scale = target_total / current_total
        scaled = np.round(alloc * scale).astype(int)
        # Adjust for rounding errors to hit exact total
        diff = target_total - scaled.sum()
        if diff != 0:
            # Add/subtract from largest cell
            max_idx = np.unravel_index(np.argmax(scaled), scaled.shape)
            scaled[max_idx] += diff
        return scaled

    def test_scaling_preserves_v_approximately(self):
        """Scaling should preserve Cramér's V within tolerance."""
        for v_target, alloc in CAMELYON_PAPER_ALLOCATIONS.items():
            if v_target in (0.0, 1.0):
                # Edge cases: V=0 and V=1 are exact
                continue
            with self.subTest(v_target=v_target):
                # Scale from 8400 to 840 (10%)
                scaled = self._scale_allocation(alloc, 840)
                v_original = cramers_v_from_counts(alloc)
                v_scaled = cramers_v_from_counts(scaled)
                # Allow larger tolerance for scaled versions due to rounding
                self.assertAlmostEqual(
                    v_scaled, v_original, delta=0.05,
                    msg=f"V={v_target}: original {v_original}, scaled {v_scaled}"
                )

    def test_scaling_preserves_total(self):
        """Scaled allocation should have exact target total."""
        alloc = CAMELYON_PAPER_ALLOCATIONS[0.00]
        for target in [840, 1000, 4200, 100]:
            with self.subTest(target=target):
                scaled = self._scale_allocation(alloc, target)
                self.assertEqual(scaled.sum(), target)


class TestSplitGenerationInvariants(unittest.TestCase):
    """Test that generated splits maintain required invariants."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic dataset mimicking CAMELYON structure."""
        rows = []
        sample_idx = 0
        # ID centers: RUMC, UMCU; OOD centers: CWZ, RST
        for label in ["normal", "tumor"]:
            for center in ["RUMC", "UMCU", "CWZ", "RST"]:
                n_slides = 10 if center in ["RUMC", "UMCU"] else 3
                for slide in range(n_slides):
                    slide_id = f"{center}_{label}_{slide}"
                    n_patches = 50 if center in ["RUMC", "UMCU"] else 20
                    for patch in range(n_patches):
                        rows.append({
                            "sample_id": f"s{sample_idx}",
                            "image_path": f"/fake/{sample_idx}.png",
                            "label": label,
                            "medical_center": center,
                            "slide_id": slide_id,
                        })
                        sample_idx += 1
        cls.df = pd.DataFrame(rows)
        cls.id_centers = ["RUMC", "UMCU"]
        cls.ood_centers = ["CWZ", "RST"]

    def test_no_slide_leakage_between_train_and_id_test(self):
        """No slide should appear in both train and id_test partitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="test",
                repetitions=2,
                correlation_levels=[0.0, 0.5, 1.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
            )

            for split_df in splits:
                train_slides = set(
                    split_df[split_df["partition"] == "train"]["slide_id"]
                )
                id_test_slides = set(
                    split_df[split_df["partition"] == "id_test"]["slide_id"]
                )
                overlap = train_slides & id_test_slides
                self.assertEqual(
                    len(overlap), 0,
                    f"Slide leakage detected: {overlap}"
                )

    def test_fixed_test_sets_across_correlation_levels(self):
        """Within same rep, ID and OOD test sets should be identical across V levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="test",
                repetitions=1,
                correlation_levels=[0.0, 0.5, 1.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
            )

            # Group by rep
            by_rep = {}
            for split_df in splits:
                rep = split_df["rep"].iloc[0]
                by_rep.setdefault(rep, []).append(split_df)

            for rep, rep_splits in by_rep.items():
                # Get test sample IDs from first split
                first_id_test = set(
                    rep_splits[0][rep_splits[0]["partition"] == "id_test"]["sample_id"]
                )
                first_ood_test = set(
                    rep_splits[0][rep_splits[0]["partition"] == "ood_test"]["sample_id"]
                )

                # Compare with all other splits in same rep
                for split_df in rep_splits[1:]:
                    rho = split_df["correlation_level"].iloc[0]
                    id_test = set(
                        split_df[split_df["partition"] == "id_test"]["sample_id"]
                    )
                    ood_test = set(
                        split_df[split_df["partition"] == "ood_test"]["sample_id"]
                    )

                    self.assertEqual(
                        first_id_test, id_test,
                        f"Rep {rep}, rho={rho}: ID test set differs from rho=0"
                    )
                    self.assertEqual(
                        first_ood_test, ood_test,
                        f"Rep {rep}, rho={rho}: OOD test set differs from rho=0"
                    )

    def test_deterministic_generation(self):
        """Same seed should produce identical splits."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            splits_a = generate_apd_splits(
                df=self.df,
                output_dir=Path(d1),
                dataset_name="test",
                repetitions=2,
                correlation_levels=[0.0, 1.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=12345,
            )
            splits_b = generate_apd_splits(
                df=self.df,
                output_dir=Path(d2),
                dataset_name="test",
                repetitions=2,
                correlation_levels=[0.0, 1.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=12345,
            )

            self.assertEqual(len(splits_a), len(splits_b))
            for a, b in zip(splits_a, splits_b):
                # Check same samples in same order
                self.assertListEqual(
                    a["sample_id"].tolist(),
                    b["sample_id"].tolist()
                )

    def test_training_totals_consistent_across_correlation_levels(self):
        """All splits within same rep should have same training set size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="test",
                repetitions=2,
                correlation_levels=[0.0, 0.5, 1.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
            )

            # Group by rep
            by_rep = {}
            for split_df in splits:
                rep = split_df["rep"].iloc[0]
                by_rep.setdefault(rep, []).append(split_df)

            for rep, rep_splits in by_rep.items():
                train_sizes = []
                for split_df in rep_splits:
                    train_size = (split_df["partition"] == "train").sum()
                    train_sizes.append(train_size)

                # All should be equal
                self.assertEqual(
                    len(set(train_sizes)), 1,
                    f"Rep {rep}: training sizes vary: {train_sizes}"
                )

    def test_realized_v_increases_with_target(self):
        """Realized Cramér's V should increase as target correlation increases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="test",
                repetitions=1,
                correlation_levels=[0.0, 0.5, 1.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
            )

            v_values = []
            for split_df in splits:
                v_realized = split_df["cramers_v_realized"].iloc[0]
                v_values.append(v_realized)

            # Should be monotonically increasing (allowing small tolerance)
            for i in range(len(v_values) - 1):
                self.assertLessEqual(
                    v_values[i], v_values[i + 1] + 0.05,
                    f"V should increase: {v_values}"
                )

    def test_only_id_centers_in_training(self):
        """Training partition should only contain ID centers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="test",
                repetitions=1,
                correlation_levels=[0.0, 1.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
            )

            for split_df in splits:
                train_centers = set(
                    split_df[split_df["partition"] == "train"]["medical_center"]
                )
                for center in train_centers:
                    self.assertIn(
                        center, self.id_centers,
                        f"OOD center {center} found in training"
                    )

    def test_ood_test_contains_only_ood_centers(self):
        """OOD test partition should only contain OOD centers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="test",
                repetitions=1,
                correlation_levels=[0.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
            )

            for split_df in splits:
                ood_centers = set(
                    split_df[split_df["partition"] == "ood_test"]["medical_center"]
                )
                for center in ood_centers:
                    self.assertIn(
                        center, self.ood_centers,
                        f"ID center {center} found in OOD test"
                    )


class TestPaperModeGeneration(unittest.TestCase):
    """Test paper-faithful split generation mode."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic dataset mimicking CAMELYON structure."""
        rows = []
        sample_idx = 0
        # Match paper's center names
        for label in ["normal", "tumor"]:
            for center in ["RUMC", "UMCU", "CWZ", "RST", "LPON"]:
                n_slides = 20 if center in ["RUMC", "UMCU"] else 5
                for slide in range(n_slides):
                    slide_id = f"{center}_{label}_{slide}"
                    n_patches = 100 if center in ["RUMC", "UMCU"] else 30
                    for patch in range(n_patches):
                        rows.append({
                            "sample_id": f"s{sample_idx}",
                            "image_path": f"/fake/{sample_idx}.png",
                            "label": label,
                            "medical_center": center,
                            "slide_id": slide_id,
                        })
                        sample_idx += 1
        cls.df = pd.DataFrame(rows)
        cls.id_centers = ["RUMC", "UMCU"]
        cls.ood_centers = ["CWZ", "RST", "LPON"]

    def test_paper_mode_uses_paper_v_levels(self):
        """Paper mode should use exact V levels from the paper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="camelyon",
                repetitions=1,
                correlation_levels=[0.0, 0.5],  # This should be overridden
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
                mode="paper",
            )

            # Should have 8 splits (paper V levels), not 2
            self.assertEqual(len(splits), 8)

            # Check V levels match paper
            v_levels = sorted(set(s["correlation_level"].iloc[0] for s in splits))
            expected = [0.00, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.00]
            self.assertEqual(v_levels, expected)

    def test_paper_mode_allocation_proportions(self):
        """Paper mode should produce allocations with correct proportions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="camelyon",
                repetitions=1,
                correlation_levels=[0.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
                mode="paper",
            )

            # Find V=0 split (balanced)
            v0_split = [s for s in splits if abs(s["correlation_level"].iloc[0]) < 0.01][0]
            train = v0_split[v0_split["partition"] == "train"]

            # For V=0, all cells should have equal counts
            ctab = pd.crosstab(train["label"], train["medical_center"])

            # Check roughly equal distribution
            counts = ctab.values.flatten()
            # Allow 10% deviation due to rounding
            mean_count = counts.mean()
            for c in counts:
                self.assertAlmostEqual(c, mean_count, delta=mean_count * 0.15)

    def test_paper_mode_v1_diagonal_pattern(self):
        """Paper mode at V=1 should show diagonal association pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="camelyon",
                repetitions=1,
                correlation_levels=[0.0],  # Will be overridden
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
                mode="paper",
            )

            # Find V=1 split
            v1_split = [s for s in splits if abs(s["correlation_level"].iloc[0] - 1.0) < 0.01][0]
            train = v1_split[v1_split["partition"] == "train"]

            # At V=1: RUMC should be mostly tumor, UMCU should be mostly normal
            ctab = pd.crosstab(train["label"], train["medical_center"])

            # Paper pattern: normal-RUMC=0, normal-UMCU=high, tumor-RUMC=high, tumor-UMCU=0
            # Check the diagonal pattern (allowing for capacity constraints)
            if "RUMC" in ctab.columns and "UMCU" in ctab.columns:
                if "normal" in ctab.index and "tumor" in ctab.index:
                    # Off-diagonal should be much smaller than diagonal
                    normal_rumc = ctab.loc["normal", "RUMC"]
                    tumor_umcu = ctab.loc["tumor", "UMCU"]
                    normal_umcu = ctab.loc["normal", "UMCU"]
                    tumor_rumc = ctab.loc["tumor", "RUMC"]

                    # Diagonal (normal-UMCU, tumor-RUMC) should dominate
                    self.assertGreater(
                        normal_umcu + tumor_rumc,
                        normal_rumc + tumor_umcu,
                        "V=1 should show diagonal association pattern"
                    )

    def test_paper_mode_maintains_invariants(self):
        """Paper mode should maintain all split invariants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="camelyon",
                repetitions=2,
                correlation_levels=[0.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
                mode="paper",
            )

            for split_df in splits:
                # No slide leakage
                train_slides = set(split_df[split_df["partition"] == "train"]["slide_id"])
                test_slides = set(split_df[split_df["partition"] == "id_test"]["slide_id"])
                self.assertEqual(len(train_slides & test_slides), 0)

                # Only ID centers in training
                train_centers = set(split_df[split_df["partition"] == "train"]["medical_center"])
                for c in train_centers:
                    self.assertIn(c, self.id_centers)

    def test_paper_mode_realized_v_close_to_target(self):
        """Realized Cramér's V should be close to target V."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="camelyon",
                repetitions=1,
                correlation_levels=[0.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
                mode="paper",
            )

            for split_df in splits:
                v_target = split_df["correlation_level"].iloc[0]
                v_realized = split_df["cramers_v_realized"].iloc[0]

                # Allow some deviation due to scaling and rounding
                # V=0 and V=1 should be more precise
                if v_target in (0.0, 1.0):
                    tolerance = 0.1
                else:
                    tolerance = 0.15

                self.assertAlmostEqual(
                    v_realized, v_target, delta=tolerance,
                    msg=f"Target V={v_target}, realized V={v_realized}"
                )

    def test_interpolate_mode_still_works(self):
        """Default interpolate mode should still work after adding paper mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = generate_apd_splits(
                df=self.df,
                output_dir=Path(tmpdir),
                dataset_name="camelyon",
                repetitions=1,
                correlation_levels=[0.0, 0.5, 1.0],
                id_centers=self.id_centers,
                ood_centers=self.ood_centers,
                id_test_fraction=0.2,
                seed=42,
                mode="interpolate",
            )

            # Should have 3 splits (as specified)
            self.assertEqual(len(splits), 3)

            # V levels should match input
            v_levels = [s["correlation_level"].iloc[0] for s in splits]
            self.assertEqual(v_levels, [0.0, 0.5, 1.0])


class TestCramersVEdgeCases(unittest.TestCase):
    """Test Cramér's V calculation edge cases."""

    def test_single_row(self):
        """Single row matrix."""
        counts = np.array([[50, 50]])
        v = cramers_v_from_counts(counts)
        self.assertEqual(v, 0.0)

    def test_single_column(self):
        """Single column matrix."""
        counts = np.array([[50], [50]])
        v = cramers_v_from_counts(counts)
        self.assertEqual(v, 0.0)

    def test_larger_matrix(self):
        """3x3 matrix with perfect association."""
        counts = np.array([
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 100]
        ])
        v = cramers_v_from_counts(counts)
        self.assertAlmostEqual(v, 1.0, places=5)

    def test_3x3_no_association(self):
        """3x3 matrix with no association."""
        counts = np.array([
            [33, 33, 34],
            [33, 34, 33],
            [34, 33, 33]
        ])
        v = cramers_v_from_counts(counts)
        self.assertLess(v, 0.1)


if __name__ == "__main__":
    unittest.main()
