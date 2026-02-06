from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
        self._workspace_created: bool = False

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
                # Log per-rho accuracies for interpretability
                for rho, (mean_acc, _) in apd.acc_id_by_rho.items():
                    rho_str = f"{rho:.2f}".replace(".", "_")
                    log_metrics[f"{dataset_name}/{model_name}/acc_id_rho{rho_str}"] = mean_acc
                for rho, (mean_acc, _) in apd.acc_ood_by_rho.items():
                    rho_str = f"{rho:.2f}".replace(".", "_")
                    log_metrics[f"{dataset_name}/{model_name}/acc_ood_rho{rho_str}"] = mean_acc

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

        return metric_rows, log_metrics

    def _setup_wandb_workspace(self, metric_keys: List[str]) -> None:
        """Create a wandb workspace with native LinePlot panels (once)."""
        if self._workspace_created:
            return

        try:
            import wandb
            if wandb.run is None:
                return
            import wandb_workspaces.workspaces as ws
            import wandb_workspaces.reports.v2 as wr
        except ImportError:
            logger.info(
                "[PathoROB] wandb_workspaces not installed; "
                "skipping workspace creation. Install with: "
                "pip install wandb-workspaces"
            )
            self._workspace_created = True
            return

        try:
            self._build_workspace(wandb, ws, wr, metric_keys)
            self._workspace_created = True
        except Exception as exc:
            logger.warning(
                f"[PathoROB] Failed to create wandb workspace: {exc}. "
                "Scalar metrics are unaffected."
            )

    @staticmethod
    def _inject_metric_colors(panel, wr) -> None:
        """Patch a LinePlot panel so each y-metric gets a distinct color."""
        if not isinstance(panel, wr.LinePlot) or len(panel.y) <= 1:
            return
        # Qualitative palette (colorblind-friendly, from ColorBrewer Set2 + Dark2)
        palette = [
            "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
            "#66a61e", "#e6ab02", "#a6761d", "#666666",
            "#377eb8", "#e41a1c", "#984ea3", "#ff7f00",
        ]
        colors = {key: palette[i % len(palette)] for i, key in enumerate(panel.y)}
        # Wrap _to_model to inject override_colors into the internal config
        original_to_model = panel._to_model
        def patched_to_model():
            model = original_to_model()
            model.config.override_colors = colors
            return model
        panel._to_model = patched_to_model

    def _build_workspace(self, wandb, ws, wr, metric_keys: List[str]) -> None:
        """Build and save a wandb workspace with LinePlot panels."""
        prefix = "tune/pathorob/"

        # Discover datasets, models, and metric structure from logged keys
        datasets: Dict[str, Dict[str, List[str]]] = {}
        for key in metric_keys:
            # Keys are like "camelyon/student/ri" or "camelyon/student/acc_id_rho0_00"
            parts = key.split("/")
            if len(parts) < 3:
                continue
            ds, model, metric = parts[0], parts[1], "/".join(parts[2:])
            datasets.setdefault(ds, {}).setdefault(model, []).append(metric)

        sections: List[ws.Section] = []

        for dataset in sorted(datasets):
            title_prefix = dataset.capitalize()
            panels: list = []

            # --- Group 1: Student vs Teacher comparison panels ---
            comparison_metrics = [
                ("ri", "Robustness Index (RI)"),
                ("apd_id", "APD (In-Distribution)"),
                ("apd_ood", "APD (Out-of-Distribution)"),
                ("apd_avg", "APD (Average)"),
                ("clustering_score", "Clustering Score"),
            ]

            for metric_key, metric_label in comparison_metrics:
                y_keys = []
                for model in ["student", "teacher"]:
                    full_key = f"{prefix}{dataset}/{model}/{metric_key}"
                    if (
                        model in datasets.get(dataset, {})
                        and metric_key in datasets[dataset][model]
                    ):
                        y_keys.append(full_key)
                if y_keys:
                    panels.append(
                        wr.LinePlot(
                            title=f"{title_prefix} \u2014 {metric_label}",
                            x="Step",
                            y=y_keys,
                        )
                    )

            # --- Group 2: Per-rho accuracy curves ---
            for model in ["student", "teacher"]:
                model_metrics = datasets.get(dataset, {}).get(model, [])
                for eval_type, eval_label in [("acc_id", "ID"), ("acc_ood", "OOD")]:
                    rho_prefix = f"{eval_type}_rho"
                    rho_keys = sorted(
                        f"{prefix}{dataset}/{model}/{m}"
                        for m in model_metrics
                        if m.startswith(rho_prefix)
                    )
                    if rho_keys:
                        panels.append(
                            wr.LinePlot(
                                title=f"{title_prefix} \u2014 {model.capitalize()} Accuracy ({eval_label}) by \u03c1",
                                x="Step",
                                y=rho_keys,
                            )
                        )

            if panels:
                sections.append(
                    ws.Section(
                        name=f"PathoROB \u2014 {title_prefix}",
                        panels=panels,
                        is_open=True,
                        pinned=True,
                    )
                )

        if not sections:
            return

        # Add panels to the current run's workspace instead of creating a separate one
        run_url = wandb.run.url
        # run.url points to the run page; workspace URL is the project page
        project_url = run_url.rsplit("/runs/", 1)[0]
        workspace = ws.Workspace.from_url(project_url)

        # Remove any existing PathoROB sections to avoid duplicates on re-run
        pathorob_prefix = "PathoROB"
        workspace.sections = [
            s for s in workspace.sections
            if not s.name.startswith(pathorob_prefix)
        ]

        # Inject per-metric colors into all LinePlot panels
        for section in sections:
            for panel in section.panels:
                self._inject_metric_colors(panel, wr)

        # Prepend our sections so they appear at the top
        workspace.sections = sections + workspace.sections

        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            workspace.save()
        logger.info("[PathoROB] Wandb workspace updated with PathoROB panels.")

    @torch.no_grad()
    def run(self, student: nn.Module, teacher: nn.Module, epoch: int) -> PluginResult:
        if not is_main_process():
            return PluginResult(name=self.name)

        student_backbone = self._get_backbone(student)
        teacher_backbone = self._get_backbone(teacher)
        student_backbone.eval()
        teacher_backbone.eval()

        all_rows: List[Dict[str, Any]] = []
        all_logs: Dict[str, float] = {}

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
                logger.error(f"[PathoROB] {dataset_name} failed at epoch {epoch+1}: {exc}")

        # Create wandb workspace with native LinePlot panels (once)
        if all_logs:
            self._setup_wandb_workspace(list(all_logs.keys()))

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

        return PluginResult(
            name=self.name,
            payload={"rows": all_rows},
            log_metrics=all_logs,
        )
