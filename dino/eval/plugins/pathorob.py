from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import transforms

from dino.distributed import is_main_process
from dino.eval.dataset import EvalDataset
from dino.eval.features import extract_features_single_process
from dino.eval.pathorob.apd import compute_apd
from dino.eval.pathorob.clustering import compute_clustering_score
from dino.eval.pathorob.datasets import load_manifest
from dino.eval.pathorob.ri import compute_ri
from dino.eval.pathorob.allocations import get_paper_metadata
from dino.eval.pathorob.splits import generate_apd_splits

from .base import BenchmarkPlugin, PluginResult

logger = logging.getLogger("dino")


class PathoROBPlugin(BenchmarkPlugin):
    """PathoROB robustness evaluation plugin."""

    def __init__(self, cfg: DictConfig, device: torch.device, output_dir: Path):
        self.cfg = cfg
        self.device = device
        self.name = "pathorob"
        self.output_dir = Path(output_dir)

        self.base_dir = self.output_dir / "pathorob"
        self.splits_dir = self.base_dir / "splits"
        self.metrics_dir = self.base_dir / "metrics"
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    cfg.transforms.resize,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(cfg.transforms.crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.transform_key = (
            f"resize={cfg.transforms.resize}|crop={cfg.transforms.crop_size}|"
            "norm=imagenet"
        )
        logging.info(f"[PathoROB] Using transforms: {self.transform_key}")

        self._manifest_cache: Dict[str, pd.DataFrame] = {}
        self._apd_split_cache: Dict[str, List[pd.DataFrame]] = {}
        self._metric_history: Dict[str, List[Tuple[int, float]]] = {}

    def should_run(self, epoch: int) -> bool:
        return ((epoch + 1) % int(self.cfg.tune_every)) == 0

    def _get_backbone(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "backbone"):
            return model.backbone
        return model

    def _load_manifest(self, dataset_name: str, csv_path: str) -> pd.DataFrame:
        if dataset_name in self._manifest_cache:
            return self._manifest_cache[dataset_name]
        df = load_manifest(csv_path, dataset_name)
        self._manifest_cache[dataset_name] = df
        return df

    def _extract_features(
        self,
        student_backbone: nn.Module,
        teacher_backbone: nn.Module,
        manifest_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        dataset = EvalDataset(
            manifest_df,
            transform=self.transform,
            image_col="image_path",
            label_col="label",
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=int(self.cfg.batch_size_per_gpu),
            num_workers=int(self.cfg.num_workers),
            pin_memory=True,
            drop_last=False,
        )

        feats, _ = extract_features_single_process(
            student_backbone,
            teacher_backbone,
            loader,
            self.device,
        )

        out: Dict[str, np.ndarray] = {}
        for model_name in ["student", "teacher"]:
            tensor = feats[model_name]
            if tensor is None:
                raise RuntimeError(f"Feature extraction returned None for {model_name}")
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().cpu().numpy().astype(np.float32)
            else:
                arr = np.asarray(tensor, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            out[model_name] = arr / norms
        return out

    def _get_split_params(self, dataset_cfg: DictConfig) -> Dict[str, Any]:
        """Get current split generation parameters as a dict for comparison."""
        return {
            "repetitions": int(self.cfg.apd.repetitions),
            "id_test_fraction": float(self.cfg.apd.id_test_fraction),
            "seed": int(self.cfg.seed),
            "mode": str(self.cfg.apd.get("mode", "paper")),
            "id_centers": sorted(list(dataset_cfg.id_centers)),
            "ood_centers": sorted(list(dataset_cfg.ood_centers)),
            # Only include correlation_levels if mode is custom
            "correlation_levels": (
                sorted(list(self.cfg.apd.correlation_levels))
                if self.cfg.apd.get("mode", "paper") == "custom"
                else None
            ),
        }

    def _validate_paper_mode_centers(self, dataset_name: str, dataset_cfg: DictConfig) -> None:
        """Validate that config centers match paper specification when using paper mode."""
        mode = str(self.cfg.apd.get("mode", "paper"))
        if mode != "paper":
            return

        try:
            paper_classes, paper_id, paper_ood = get_paper_metadata(dataset_name)
        except ValueError:
            # Dataset not in paper, can't validate
            return

        config_id = sorted(list(dataset_cfg.id_centers))
        config_ood = sorted(list(dataset_cfg.ood_centers))

        if config_id != sorted(paper_id):
            raise ValueError(
                f"[APD] Paper mode requires id_centers={paper_id} for {dataset_name}, "
                f"but config has {config_id}. Use mode='custom' for custom centers."
            )
        if config_ood != sorted(paper_ood):
            raise ValueError(
                f"[APD] Paper mode requires ood_centers={paper_ood} for {dataset_name}, "
                f"but config has {config_ood}. Use mode='custom' for custom centers."
            )

    def _ensure_apd_splits(self, dataset_name: str, dataset_cfg: DictConfig, manifest_df: pd.DataFrame) -> List[pd.DataFrame]:
        if dataset_name in self._apd_split_cache:
            return self._apd_split_cache[dataset_name]

        # Validate centers match paper spec when in paper mode
        self._validate_paper_mode_centers(dataset_name, dataset_cfg)

        split_dir = self.splits_dir / dataset_name
        metadata_file = split_dir / "split_params.json"
        split_files = sorted(split_dir.glob("rep_*/split_*.csv"))

        current_params = self._get_split_params(dataset_cfg)
        need_regenerate = True

        if split_files and metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    saved_params = json.load(f)
                if saved_params == current_params:
                    need_regenerate = False
                    logger.info(f"[APD] Loading existing splits for {dataset_name} (params match)")
                else:
                    logger.info(f"[APD] Regenerating splits for {dataset_name} (params changed)")
                    logger.debug(f"[APD] Old params: {saved_params}")
                    logger.debug(f"[APD] New params: {current_params}")
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"[APD] Could not read split metadata, regenerating splits")
        elif split_files:
            logger.warning(f"[APD] Found splits but no metadata file, regenerating to ensure consistency")

        splits: List[pd.DataFrame] = []
        if not need_regenerate:
            for p in split_files:
                splits.append(pd.read_csv(p))
        else:
            splits = generate_apd_splits(
                df=manifest_df,
                output_dir=self.splits_dir,
                dataset_name=dataset_name,
                repetitions=int(self.cfg.apd.repetitions),
                correlation_levels=list(self.cfg.apd.get("correlation_levels", [])),
                id_centers=list(dataset_cfg.id_centers),
                ood_centers=list(dataset_cfg.ood_centers),
                id_test_fraction=float(self.cfg.apd.id_test_fraction),
                seed=int(self.cfg.seed),
                mode=str(self.cfg.apd.get("mode", "paper")),
            )
            # Save metadata for future runs
            split_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, "w") as f:
                json.dump(current_params, f, indent=2)

        self._apd_split_cache[dataset_name] = splits
        return splits

    def _record(self, key: str, epoch: int, value: float) -> None:
        """Append a (epoch, value) pair to the metric history."""
        self._metric_history.setdefault(key, []).append((epoch, value))

    def _run_dataset(
        self,
        dataset_name: str,
        dataset_cfg: DictConfig,
        student_backbone: nn.Module,
        teacher_backbone: nn.Module,
        epoch: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        manifest_df = self._load_manifest(dataset_name, dataset_cfg.manifest_csv)
        features_by_model = self._extract_features(
            student_backbone,
            teacher_backbone,
            manifest_df,
        )
        sample_to_idx = {sid: i for i, sid in enumerate(manifest_df["sample_id"].tolist())}

        metric_rows: List[Dict[str, Any]] = []
        log_metrics: Dict[str, float] = {}

        for model_name in ["student", "teacher"]:
            model_feats = features_by_model[model_name]

            if bool(self.cfg.ri.enable):
                fixed_k = int(self.cfg.ri.fixed_k.get(dataset_name, self.cfg.ri.default_k))
                ri = compute_ri(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    features=model_feats,
                    manifest_df=manifest_df,
                    policy=str(self.cfg.ri.k_selection_policy),
                    fixed_k=fixed_k,
                    k_candidates=list(self.cfg.ri.k_candidates),
                    max_pairs=(None if self.cfg.get("max_pairs", 0) <= 0 else int(self.cfg.get("max_pairs", 0))),
                    random_state=int(self.cfg.seed) + epoch,
                )
                metric_rows.append(
                    {
                        "plugin": self.name,
                        "dataset": dataset_name,
                        "model": model_name,
                        "metric": "ri",
                        "epoch": int(epoch + 1),
                        "value": float(ri.value),
                        "std": float(ri.std),
                        "n": int(ri.n_pairs),
                        "extra": json.dumps({"k": int(ri.k)}),
                    }
                )
                log_metrics[f"{dataset_name}/{model_name}/ri"] = float(ri.value)
                self._record(f"{dataset_name}/{model_name}/ri", epoch, float(ri.value))

            if bool(self.cfg.apd.enable):
                split_frames = self._ensure_apd_splits(dataset_name, dataset_cfg, manifest_df)
                aligned_splits: List[pd.DataFrame] = []
                for sp in split_frames:
                    s = sp.copy()
                    s["feature_index"] = s["sample_id"].map(sample_to_idx)
                    s = s[s["feature_index"].notna()].copy()
                    s["feature_index"] = s["feature_index"].astype(int)
                    aligned_splits.append(s)

                apd = compute_apd(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    features=model_feats,
                    all_splits=aligned_splits,
                    seed=int(self.cfg.seed) + epoch,
                )

                metric_rows.extend(
                    [
                        {
                            "plugin": self.name,
                            "dataset": dataset_name,
                            "model": model_name,
                            "metric": "apd_id",
                            "epoch": int(epoch + 1),
                            "value": float(apd.apd_id),
                            "std": float(apd.apd_id_std),
                            "n": int(self.cfg.apd.repetitions),
                            "extra": "{}",
                        },
                        {
                            "plugin": self.name,
                            "dataset": dataset_name,
                            "model": model_name,
                            "metric": "apd_ood",
                            "epoch": int(epoch + 1),
                            "value": float(apd.apd_ood),
                            "std": float(apd.apd_ood_std),
                            "n": int(self.cfg.apd.repetitions),
                            "extra": "{}",
                        },
                        {
                            "plugin": self.name,
                            "dataset": dataset_name,
                            "model": model_name,
                            "metric": "apd_avg",
                            "epoch": int(epoch + 1),
                            "value": float(apd.apd_avg),
                            "std": float(apd.apd_avg_std),
                            "n": int(self.cfg.apd.repetitions),
                            "extra": "{}",
                        },
                    ]
                )
                log_metrics[f"{dataset_name}/{model_name}/apd_id"] = float(apd.apd_id)
                log_metrics[f"{dataset_name}/{model_name}/apd_ood"] = float(apd.apd_ood)
                log_metrics[f"{dataset_name}/{model_name}/apd_avg"] = float(apd.apd_avg)
                self._record(f"{dataset_name}/{model_name}/apd_id", epoch, float(apd.apd_id))
                self._record(f"{dataset_name}/{model_name}/apd_ood", epoch, float(apd.apd_ood))
                self._record(f"{dataset_name}/{model_name}/apd_avg", epoch, float(apd.apd_avg))
                # Log per-rho accuracies for interpretability
                for rho, (mean_acc, _) in apd.acc_id_by_rho.items():
                    rho_str = f"{rho:.2f}".replace(".", "_")
                    log_metrics[f"{dataset_name}/{model_name}/acc_id_rho{rho_str}"] = mean_acc
                    self._record(f"{dataset_name}/{model_name}/acc_id/rho{rho_str}", epoch, mean_acc)
                for rho, (mean_acc, _) in apd.acc_ood_by_rho.items():
                    rho_str = f"{rho:.2f}".replace(".", "_")
                    log_metrics[f"{dataset_name}/{model_name}/acc_ood_rho{rho_str}"] = mean_acc
                    self._record(f"{dataset_name}/{model_name}/acc_ood/rho{rho_str}", epoch, mean_acc)

            if bool(self.cfg.clustering.enable):
                cl = compute_clustering_score(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    features=model_feats,
                    manifest_df=manifest_df,
                    repeats=int(self.cfg.clustering.repeats),
                    k_min=int(self.cfg.clustering.k_min),
                    k_max=int(self.cfg.clustering.k_max),
                    max_pairs=(None if self.cfg.get("max_pairs", 0) <= 0 else int(self.cfg.get("max_pairs", 0))),
                    random_state=int(self.cfg.seed) + epoch,
                )
                metric_rows.append(
                    {
                        "plugin": self.name,
                        "dataset": dataset_name,
                        "model": model_name,
                        "metric": "clustering_score",
                        "epoch": int(epoch + 1),
                        "value": float(cl.score),
                        "std": float(cl.std),
                        "n": int(cl.n_pairs),
                        "extra": "{}",
                    }
                )
                log_metrics[f"{dataset_name}/{model_name}/clustering_score"] = float(cl.score)
                self._record(f"{dataset_name}/{model_name}/clustering_score", epoch, float(cl.score))

        return metric_rows, log_metrics

    def _get_history(self, key: str) -> Tuple[List[int], List[float]]:
        """Return sorted (epochs, values) lists for a history key."""
        pairs = self._metric_history.get(key, [])
        pairs_sorted = sorted(pairs, key=lambda p: p[0])
        epochs = [p[0] for p in pairs_sorted]
        values = [p[1] for p in pairs_sorted]
        return epochs, values

    def _build_wandb_panels(self) -> Dict[str, Any]:
        """Build wandb.plot.line_series panels from accumulated metric history."""
        try:
            import wandb
            if wandb.run is None:
                return {}
        except ImportError:
            return {}

        panels: Dict[str, Any] = {}

        # Discover datasets that have history entries
        datasets = set()
        for key in self._metric_history:
            # Keys are like "camelyon/student/ri"
            parts = key.split("/")
            if len(parts) >= 3:
                datasets.add(parts[0])

        for dataset in sorted(datasets):
            title_prefix = dataset.capitalize()

            # --- Group 1: Student vs Teacher comparison panels ---
            comparison_metrics = [
                ("ri", "Robustness Index (RI)", "RI"),
                ("apd_id", "APD (In-Distribution)", "APD"),
                ("apd_ood", "APD (Out-of-Distribution)", "APD"),
                ("apd_avg", "APD (Average)", "APD"),
                ("clustering_score", "Clustering Score", "Score"),
            ]

            for metric_key, metric_label, y_label in comparison_metrics:
                s_epochs, s_values = self._get_history(f"{dataset}/student/{metric_key}")
                t_epochs, t_values = self._get_history(f"{dataset}/teacher/{metric_key}")
                if not s_epochs and not t_epochs:
                    continue
                panel_key = f"panels/{dataset}_{metric_key}"
                panels[panel_key] = wandb.plot.line_series(
                    xs=[s_epochs, t_epochs],
                    ys=[s_values, t_values],
                    keys=["student", "teacher"],
                    title=f"{title_prefix} \u2014 {metric_label}",
                    xname="Epoch",
                )

            # --- Group 2: Per-rho accuracy curves ---
            for model_name in ["student", "teacher"]:
                for eval_type, eval_label in [("acc_id", "ID"), ("acc_ood", "OOD")]:
                    # Find all rho keys for this dataset/model/eval_type
                    prefix = f"{dataset}/{model_name}/{eval_type}/rho"
                    rho_keys = sorted(
                        [k for k in self._metric_history if k.startswith(prefix)]
                    )
                    if not rho_keys:
                        continue

                    xs_list = []
                    ys_list = []
                    legend_keys = []
                    for rho_key in rho_keys:
                        # Extract rho value: "camelyon/student/acc_id/rho0_14" -> "0_14" -> "0.14"
                        rho_suffix = rho_key[len(prefix):]
                        rho_display = f"\u03c1={rho_suffix.replace('_', '.')}"
                        epochs, values = self._get_history(rho_key)
                        xs_list.append(epochs)
                        ys_list.append(values)
                        legend_keys.append(rho_display)

                    panel_key = f"panels/{dataset}_{eval_type}_{model_name}"
                    panels[panel_key] = wandb.plot.line_series(
                        xs=xs_list,
                        ys=ys_list,
                        keys=legend_keys,
                        title=f"{title_prefix} \u2014 {model_name.capitalize()} Accuracy ({eval_label}) by \u03c1",
                        xname="Epoch",
                    )

        return panels

    @torch.no_grad()
    def run(self, student: nn.Module, teacher: nn.Module, epoch: int) -> PluginResult:
        if not is_main_process():
            return PluginResult(name=self.name)

        student_backbone = self._get_backbone(student)
        teacher_backbone = self._get_backbone(teacher)
        student_backbone.eval()
        teacher_backbone.eval()

        all_rows: List[Dict[str, Any]] = []
        all_logs: Dict[str, Any] = {}

        # add an assert to make sure that at least one dataset is enabled
        any_enabled = any(
            bool(dataset_cfg.enable) for dataset_cfg in self.cfg.datasets.values()
        )
        if not any_enabled:
            logging.warning("[PathoROB] No datasets are enabled for evaluation.")
        for dataset_name, dataset_cfg in self.cfg.datasets.items():
            if not bool(dataset_cfg.enable):
                continue
            try:
                rows, logs = self._run_dataset(
                    dataset_name=dataset_name,
                    dataset_cfg=dataset_cfg,
                    student_backbone=student_backbone,
                    teacher_backbone=teacher_backbone,
                    epoch=epoch,
                )
                all_rows.extend(rows)
                all_logs.update(logs)
            except Exception as exc:
                err_file = self.metrics_dir / f"epoch_{epoch+1:04d}_{dataset_name}_error.txt"
                err_file.write_text(traceback.format_exc())
                logging.info(f"[PathoROB] {dataset_name} failed at epoch {epoch+1}: {exc}")

        # Build wandb line_series panels from accumulated history
        all_logs.update(self._build_wandb_panels())

        if all_rows:
            df = pd.DataFrame(all_rows)
            out_csv = self.metrics_dir / f"epoch_{epoch+1:04d}.csv"
            out_json = self.metrics_dir / f"epoch_{epoch+1:04d}.json"
            df.to_csv(out_csv, index=False)
            out_json.write_text(json.dumps(all_rows, indent=2))

            roll_path = self.metrics_dir / "all_metrics.csv"
            if roll_path.exists():
                old = pd.read_csv(roll_path)
                pd.concat([old, df], axis=0).reset_index(drop=True).to_csv(roll_path, index=False)
            else:
                df.to_csv(roll_path, index=False)

        return PluginResult(name=self.name, payload={"rows": all_rows}, log_metrics=all_logs)
