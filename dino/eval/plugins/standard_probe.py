from __future__ import annotations
import logging
import warnings

import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import pandas as pd

from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig
from torchvision import transforms

from dino.distributed import is_main_process, is_enabled_and_multiple_gpus

logger = logging.getLogger("dino")
from dino.eval.dataset import EvalDataset
from dino.eval.features import extract_multiple_features
from dino.eval.evaluators import KNNEvaluator, LinearEvaluator

from .base import BenchmarkPlugin, PluginResult


class StandardProbePlugin(BenchmarkPlugin):
    """Legacy train/test benchmark evaluations (KNN or linear probe)."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.name = "standard_probe"
        self.distributed = is_enabled_and_multiple_gpus()
        self.transform_key = (
            f"resize={cfg.transforms.resize}|crop={cfg.transforms.crop_size}|"
            "norm=imagenet"
        )

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

        logger.info(f"[StandardProbe] Using transforms: {self.transform_key}")

        self.evaluators = {}
        self.primary_benchmark = None
        for bench in cfg.benchmarks:
            name = bench.name
            if bench.evaluator == "knn":
                knn_cfg = bench.get("knn", {})
                self.evaluators[name] = KNNEvaluator(
                    k=knn_cfg.get("k", 20),
                    temperature=knn_cfg.get("temperature", 0.07),
                )
            elif bench.evaluator == "linear":
                linear_cfg = bench.get("linear", {})
                self.evaluators[name] = LinearEvaluator(
                    epochs=linear_cfg.get("epochs", 100),
                    lr=linear_cfg.get("lr", 0.01),
                    batch_size=linear_cfg.get("batch_size", 256),
                )
            else:
                raise ValueError(f"Unknown evaluator type: {bench.evaluator}")

            if bench.get("primary", False):
                if not self.primary_benchmark is None:
                    warnings.warn(
                        f"Multiple primary benchmarks specified: "
                        f"{self.primary_benchmark} and {name}. "
                        "Only the first one will be used as primary."
                    )
                else:
                    self.primary_benchmark = name

    def should_run(self, epoch: int) -> bool:
        return ((epoch + 1) % int(self.cfg.tune_every)) == 0

    def _get_backbone(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "backbone"):
            return model.backbone
        return model

    def _load_benchmark_data(
        self, bench_cfg: DictConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
        df = pd.read_csv(bench_cfg.csv_path)

        train_df = df[df[self.cfg.partition_col] == "train"].copy()
        test_df = df[df[self.cfg.partition_col] == "test"].copy()

        train_dataset = EvalDataset(
            train_df,
            transform=self.transform,
            image_col=self.cfg.image_path_col,
            label_col=self.cfg.label_col,
        )
        label_encoder = train_dataset.label_encoder
        return train_df, test_df, label_encoder

    def _create_dataloader(
        self, df: pd.DataFrame, label_encoder: Optional[Any]
    ) -> torch.utils.data.DataLoader:
        dataset = EvalDataset(
            df,
            transform=self.transform,
            image_col=self.cfg.image_path_col,
            label_col=self.cfg.label_col,
            label_encoder=label_encoder,
        )

        if self.distributed:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.cfg.batch_size_per_gpu,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @torch.no_grad()
    def run(self, student: nn.Module, teacher: nn.Module, epoch: int) -> PluginResult:
        student_backbone = self._get_backbone(student)
        teacher_backbone = self._get_backbone(teacher)
        student_backbone.eval()
        teacher_backbone.eval()

        results = {}
        log_metrics: Dict[str, float] = {}

        for bench_cfg in self.cfg.benchmarks:
            name = bench_cfg.name

            if is_main_process():
                tqdm.tqdm.write(f"Evaluating benchmark: {name}")

            train_df, test_df, label_encoder = self._load_benchmark_data(bench_cfg)
            train_loader = self._create_dataloader(train_df, label_encoder)
            test_loader = self._create_dataloader(test_df, label_encoder)
            if label_encoder is not None:
                train_labels = torch.tensor(
                    label_encoder.transform(train_df[self.cfg.label_col].values)
                ).long()
                test_labels = torch.tensor(
                    label_encoder.transform(test_df[self.cfg.label_col].values)
                ).long()
            else:
                train_labels = torch.tensor(train_df[self.cfg.label_col].astype(int).values).long()
                test_labels = torch.tensor(test_df[self.cfg.label_col].astype(int).values).long()

            train_features, train_labels = extract_multiple_features(
                student_backbone, teacher_backbone, train_loader, self.device
            )
            test_features, test_labels = extract_multiple_features(
                student_backbone, teacher_backbone, test_loader, self.device
            )

            if is_main_process():
                for model_name in ["student", "teacher"]:
                    train_features[model_name] = nn.functional.normalize(
                        train_features[model_name], dim=1, p=2
                    )
                    test_features[model_name] = nn.functional.normalize(
                        test_features[model_name], dim=1, p=2
                    )

            if is_main_process():
                evaluator = self.evaluators[name]
                bench_results = {}

                for model_name in ["student", "teacher"]:
                    metrics = evaluator.evaluate(
                        train_features[model_name],
                        train_labels.to(self.device),
                        test_features[model_name],
                        test_labels.to(self.device),
                    )
                    bench_results[model_name] = metrics
                    for metric_name, value in metrics.items():
                        log_metrics[f"{name}/{model_name}/{metric_name}"] = float(value)

                    tqdm.tqdm.write(
                        f"  {model_name}: acc={metrics['accuracy']:.2f}% "
                        f"bal_acc={metrics['balanced_accuracy']:.2f}% "
                        f"auc={metrics['auc']:.4f}"
                    )

                results[name] = bench_results

            if self.distributed:
                dist.barrier()

        primary = None
        if self.primary_benchmark is not None:
            primary = results.get(self.primary_benchmark)

        return PluginResult(
            name=self.name,
            payload=results,
            log_metrics=log_metrics,
            primary=primary,
        )
