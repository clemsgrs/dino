import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dino.eval.pathorob.clustering import clustering_score
from dino.eval.pathorob.ri import _normalized_ri_from_neighbors
from dino.eval.pathorob.splits import cramers_v_from_counts, generate_apd_splits


class TestPathoROBProtocol(unittest.TestCase):
    def test_ri_formula(self):
        # 4 samples, 2 classes, 2 centers, one nearest neighbor each
        labels = np.array([0, 0, 1, 1])
        centers = np.array([0, 1, 0, 1])
        # neighbor table chosen so every neighbor is SO (same label, other center)
        neigh = np.array([[1], [0], [3], [2]], dtype=int)
        ri = _normalized_ri_from_neighbors(labels, centers, neigh, k=1)
        self.assertAlmostEqual(ri, 1.0, places=6)

    def test_clustering_score_formula(self):
        pred = np.array([0, 0, 1, 1])
        bio = np.array([0, 0, 1, 1])
        center = np.array([0, 1, 0, 1])
        score = clustering_score(pred, bio, center)
        self.assertGreater(score, 0.9)

    def test_cramers_v_bounds(self):
        counts = np.array([[10, 0], [0, 10]])
        v = cramers_v_from_counts(counts)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_apd_split_generation_determinism_and_integrity(self):
        rows = []
        sample_idx = 0
        # id centers: A,B ; ood center: C
        for label in ["x", "y"]:
            for center in ["A", "B", "C"]:
                for slide in range(8):
                    slide_id = f"{center}_{label}_{slide}"
                    for patch in range(5):
                        rows.append(
                            {
                                "sample_id": f"s{sample_idx}",
                                "image_path": f"/tmp/fake/{sample_idx}.png",
                                "label": label,
                                "medical_center": center,
                                "slide_id": slide_id,
                            }
                        )
                        sample_idx += 1

        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            out1 = generate_apd_splits(
                df=df,
                output_dir=Path(d1),
                dataset_name="toy",
                repetitions=2,
                correlation_levels=[0.0, 0.5, 1.0],
                id_centers=["A", "B"],
                ood_centers=["C"],
                id_test_fraction=0.25,
                seed=123,
            )
            out2 = generate_apd_splits(
                df=df,
                output_dir=Path(d2),
                dataset_name="toy",
                repetitions=2,
                correlation_levels=[0.0, 0.5, 1.0],
                id_centers=["A", "B"],
                ood_centers=["C"],
                id_test_fraction=0.25,
                seed=123,
            )

            self.assertEqual(len(out1), len(out2))
            for a, b in zip(out1, out2):
                # deterministic generation
                self.assertTrue(a[["sample_id", "partition", "rep", "split_id"]].equals(b[["sample_id", "partition", "rep", "split_id"]]))

                # leakage check
                train_slides = set(a[a["partition"] == "train"]["slide_id"].tolist())
                id_slides = set(a[a["partition"] == "id_test"]["slide_id"].tolist())
                self.assertEqual(len(train_slides.intersection(id_slides)), 0)


if __name__ == "__main__":
    unittest.main()
