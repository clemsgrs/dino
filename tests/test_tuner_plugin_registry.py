import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dino.eval.tuner import Tuner
from dino.eval.plugins.pathorob import PathoROBPlugin


class TestTunerPluginRegistry(unittest.TestCase):
    def test_unknown_plugin_type_raises(self):
        cfg = OmegaConf.create(
            {
                "plugins": [
                    {
                        "type": "unknown_plugin",
                        "enable": True,
                    }
                ],
                "pathorob": {"enable": False},
                "eva": {"enable": False},
            }
        )
        with self.assertRaises(ValueError):
            Tuner(cfg, torch.device("cpu"), output_dir=Path("/tmp"))


class TestPathoROBPaperModeValidation(unittest.TestCase):
    def test_paper_mode_rejects_wrong_id_centers(self):
        """Paper mode should reject non-paper ID centers."""
        cfg = OmegaConf.create({
            "tune_every": 1,
            "seed": 42,
            "transforms": {"resize": 256, "crop_size": 224},
            "num_workers": 0,
            "batch_size_per_gpu": 32,
            "ri": {"enable": False},
            "apd": {
                "enable": True,
                "mode": "paper",
                "repetitions": 1,
                "id_test_fraction": 0.2,
                "correlation_levels": [],
            },
            "clustering": {"enable": False},
            "datasets": {
                "camelyon": {
                    "enable": True,
                    "manifest_csv": "/tmp/fake.csv",
                    "id_centers": ["WRONG_CENTER"],  # Not paper centers
                    "ood_centers": ["CWZ", "RST", "LPON"],
                }
            },
        })
        plugin = PathoROBPlugin(cfg, torch.device("cpu"), output_dir=Path("/tmp"))
        dataset_cfg = cfg.datasets.camelyon

        with self.assertRaises(ValueError) as ctx:
            plugin._validate_paper_mode_centers("camelyon", dataset_cfg)

        self.assertIn("paper mode requires", str(ctx.exception).lower())
        self.assertIn("id_centers", str(ctx.exception).lower())

    def test_paper_mode_rejects_wrong_ood_centers(self):
        """Paper mode should reject non-paper OOD centers."""
        cfg = OmegaConf.create({
            "tune_every": 1,
            "seed": 42,
            "transforms": {"resize": 256, "crop_size": 224},
            "num_workers": 0,
            "batch_size_per_gpu": 32,
            "ri": {"enable": False},
            "apd": {
                "enable": True,
                "mode": "paper",
                "repetitions": 1,
                "id_test_fraction": 0.2,
                "correlation_levels": [],
            },
            "clustering": {"enable": False},
            "datasets": {
                "camelyon": {
                    "enable": True,
                    "manifest_csv": "/tmp/fake.csv",
                    "id_centers": ["RUMC", "UMCU"],  # Correct
                    "ood_centers": ["WRONG"],  # Not paper centers
                }
            },
        })
        plugin = PathoROBPlugin(cfg, torch.device("cpu"), output_dir=Path("/tmp"))
        dataset_cfg = cfg.datasets.camelyon

        with self.assertRaises(ValueError) as ctx:
            plugin._validate_paper_mode_centers("camelyon", dataset_cfg)

        self.assertIn("paper mode requires", str(ctx.exception).lower())
        self.assertIn("ood_centers", str(ctx.exception).lower())

    def test_paper_mode_accepts_correct_centers(self):
        """Paper mode should accept correct paper centers."""
        cfg = OmegaConf.create({
            "tune_every": 1,
            "seed": 42,
            "transforms": {"resize": 256, "crop_size": 224},
            "num_workers": 0,
            "batch_size_per_gpu": 32,
            "ri": {"enable": False},
            "apd": {
                "enable": True,
                "mode": "paper",
                "repetitions": 1,
                "id_test_fraction": 0.2,
                "correlation_levels": [],
            },
            "clustering": {"enable": False},
            "datasets": {
                "camelyon": {
                    "enable": True,
                    "manifest_csv": "/tmp/fake.csv",
                    "id_centers": ["RUMC", "UMCU"],
                    "ood_centers": ["CWZ", "RST", "LPON"],
                }
            },
        })
        plugin = PathoROBPlugin(cfg, torch.device("cpu"), output_dir=Path("/tmp"))
        dataset_cfg = cfg.datasets.camelyon

        # Should not raise
        plugin._validate_paper_mode_centers("camelyon", dataset_cfg)

    def test_interpolate_mode_allows_any_centers(self):
        """Interpolate mode should allow any centers."""
        cfg = OmegaConf.create({
            "tune_every": 1,
            "seed": 42,
            "transforms": {"resize": 256, "crop_size": 224},
            "num_workers": 0,
            "batch_size_per_gpu": 32,
            "ri": {"enable": False},
            "apd": {
                "enable": True,
                "mode": "interpolate",
                "repetitions": 1,
                "id_test_fraction": 0.2,
                "correlation_levels": [0.0, 0.5, 1.0],
            },
            "clustering": {"enable": False},
            "datasets": {
                "camelyon": {
                    "enable": True,
                    "manifest_csv": "/tmp/fake.csv",
                    "id_centers": ["ANY", "CENTERS"],
                    "ood_centers": ["WORK", "HERE"],
                }
            },
        })
        plugin = PathoROBPlugin(cfg, torch.device("cpu"), output_dir=Path("/tmp"))
        dataset_cfg = cfg.datasets.camelyon

        # Should not raise
        plugin._validate_paper_mode_centers("camelyon", dataset_cfg)


if __name__ == "__main__":
    unittest.main()
