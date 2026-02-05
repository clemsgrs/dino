from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist
import pandas as pd

from pathlib import Path
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

from dino.distributed import is_enabled_and_multiple_gpus
from .plugins import BenchmarkPlugin, PathoROBPlugin, StandardProbePlugin
import warnings


class Tuner:
    """
    Single orchestrator for evaluation plugins.
    """

    def __init__(self, cfg: DictConfig, device: torch.device, output_dir: Path):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.distributed = is_enabled_and_multiple_gpus()
        self.metrics_dir = self.output_dir / "metrics"
        self.cache_dir = self.output_dir / "cache"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.plugins: List[BenchmarkPlugin] = []
        self.primary_plugin_name: Optional[str] = None
        self._register_plugins()

        for plugin in self.plugins:
            plugin.bind_runtime(self.output_dir)

    def _register_plugins(self) -> None:
        plugin_cfgs = list(self.cfg.get("plugins", []))
        if plugin_cfgs:
            for p in plugin_cfgs:
                if not p.get("enable", True):
                    continue
                ptype = str(p.get("type", "")).strip().lower()
                if ptype == "standard_probe":
                    plugin = StandardProbePlugin(p, self.device)
                    self.plugins.append(plugin)
                    if p.get("primary", False):
                        self.primary_plugin_name = plugin.name
                elif ptype == "pathorob":
                    plugin = PathoROBPlugin(p, self.device, self.output_dir)
                    self.plugins.append(plugin)
                    if p.get("primary", False):
                        warnings.warn("PathoROB plugin cannot be primary")
                else:
                    raise ValueError(f"Unknown tuning plugin type: {ptype}")

    def _get_backbone(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "backbone"):
            return model.backbone
        return model

    @torch.no_grad()
    def tune(
        self,
        student: nn.Module,
        teacher: nn.Module,
        epoch: int,
    ) -> Dict[str, Any]:
        """Run all due plugins and return a consolidated payload."""
        # Keep models in eval mode during plugin runs
        self._get_backbone(student).eval()
        self._get_backbone(teacher).eval()

        out: Dict[str, Any] = {
            "plugins": {},
            "log_metrics": {},
            "primary": None,
        }

        for plugin in self.plugins:
            if not plugin.should_run(epoch):
                continue

            result = plugin.run(student, teacher, epoch)
            out["plugins"][plugin.name] = result.payload

            for k, v in result.log_metrics.items():
                out["log_metrics"][f"{plugin.name}/{k}"] = float(v)

            if out["primary"] is None and result.primary is not None:
                out["primary"] = result.primary

            if self.distributed:
                dist.barrier()

        self._persist_unified_metrics(results=out, epoch=epoch)
        return out

    def get_primary_results(
        self,
        results: Dict[str, Any],
    ) -> Optional[Dict[str, Dict[str, float]]]:
        return results.get("primary")

    def get_log_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        return results.get("log_metrics", {})

    def _persist_unified_metrics(self, results: Dict[str, Any], epoch: int) -> None:
        if self.metrics_dir is None:
            return

        rows: List[Dict[str, Any]] = []
        plugins_payload = results.get("plugins", {})
        for plugin_name, payload in plugins_payload.items():
            if isinstance(payload, dict) and "rows" in payload and isinstance(payload["rows"], list):
                for row in payload["rows"]:
                    if "plugin" not in row:
                        row = dict(row)
                        row["plugin"] = plugin_name
                    rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        out_csv = self.metrics_dir / f"epoch_{epoch+1:04d}.csv"
        df.to_csv(out_csv, index=False)
        roll_csv = self.metrics_dir / "all_metrics.csv"
        if roll_csv.exists():
            old = pd.read_csv(roll_csv)
            pd.concat([old, df], axis=0).reset_index(drop=True).to_csv(roll_csv, index=False)
        else:
            df.to_csv(roll_csv, index=False)
