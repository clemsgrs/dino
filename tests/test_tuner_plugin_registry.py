import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dino.eval.tuner import Tuner


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


if __name__ == "__main__":
    unittest.main()
