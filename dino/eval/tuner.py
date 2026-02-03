import sys
import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import pandas as pd

from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig
from torchvision import transforms

from .dataset import EvalDataset
from .features import extract_multiple_features
from .evaluators import KNNEvaluator, LinearEvaluator
from dino.distributed import is_main_process, is_enabled_and_multiple_gpus


class Tuner:
    """Orchestrates downstream benchmark evaluation during training.

    Evaluates both student and teacher models on multiple benchmarks using
    either KNN or linear probing.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        """
        Args:
            cfg: Tuning configuration (cfg.tuning from main config)
            device: Device to run evaluation on
        """
        self.cfg = cfg
        self.device = device
        self.distributed = is_enabled_and_multiple_gpus()

        # Build transform
        self.transform = transforms.Compose([
            transforms.Resize(
                cfg.transform.resize,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.transform.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        print("Tuner transforms")
        print(self.transform)

        # Pre-build evaluators for each benchmark
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
                self.primary_benchmark = name

    def _get_backbone(self, model: nn.Module) -> nn.Module:
        """Extract backbone from MultiCropWrapper or DDP-wrapped model."""
        # Unwrap DDP if needed
        if hasattr(model, "module"):
            model = model.module
        # MultiCropWrapper has .backbone attribute
        if hasattr(model, "backbone"):
            return model.backbone
        return model

    def _load_benchmark_data(
        self, bench_cfg: DictConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Any]]:
        """Load and split benchmark CSV into train/test DataFrames."""
        df = pd.read_csv(bench_cfg.csv_path)

        image_col = self.cfg.image_path_col
        label_col = self.cfg.label_col
        partition_col = self.cfg.partition_col

        train_df = df[df[partition_col] == "train"].copy()
        test_df = df[df[partition_col] == "test"].copy()

        # Fit label encoder on train set for consistent mapping
        train_dataset = EvalDataset(
            train_df,
            transform=self.transform,
            image_col=image_col,
            label_col=label_col,
        )
        label_encoder = train_dataset.label_encoder

        return train_df, test_df, label_encoder

    def _create_dataloader(
        self, df: pd.DataFrame, label_encoder: Optional[Any]
    ) -> torch.utils.data.DataLoader:
        """Create dataloader for a partition."""
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
    def tune(
        self,
        student: nn.Module,
        teacher: nn.Module,
        epoch: int,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Run evaluation on all benchmarks.

        Args:
            student: Student model (may be DDP-wrapped)
            teacher: Teacher model (may be DDP-wrapped)
            epoch: Current training epoch

        Returns:
            Nested dict: {benchmark_name: {model_name: {metric: value}}}
            e.g., {"benchmark_a": {"student": {"accuracy": 85.2, ...}, "teacher": {...}}}
        """
        # Get backbones (without projection heads)
        student_backbone = self._get_backbone(student)
        teacher_backbone = self._get_backbone(teacher)

        student_backbone.eval()
        teacher_backbone.eval()

        results = {}

        for bench_cfg in self.cfg.benchmarks:
            name = bench_cfg.name

            if is_main_process():
                tqdm.tqdm.write(f"Evaluating benchmark: {name}")

            # Load data
            train_df, test_df, label_encoder = self._load_benchmark_data(bench_cfg)

            # Create dataloaders
            train_loader = self._create_dataloader(train_df, label_encoder)
            test_loader = self._create_dataloader(test_df, label_encoder)

            # Extract features for train set
            train_features, train_labels = extract_multiple_features(
                student_backbone, teacher_backbone, train_loader, self.device
            )

            # Extract features for test set
            test_features, test_labels = extract_multiple_features(
                student_backbone, teacher_backbone, test_loader, self.device
            )

            # L2 normalize features (done on main process only for distributed)
            if is_main_process():
                for model_name in ["student", "teacher"]:
                    train_features[model_name] = nn.functional.normalize(
                        train_features[model_name], dim=1, p=2
                    )
                    test_features[model_name] = nn.functional.normalize(
                        test_features[model_name], dim=1, p=2
                    )

            # Run evaluation on rank 0 only
            if is_main_process():
                evaluator = self.evaluators[name]
                bench_results = {}

                for model_name in ["student", "teacher"]:
                    train_labels_device = train_labels.to(self.device)
                    test_labels_device = test_labels.to(self.device)

                    metrics = evaluator.evaluate(
                        train_features[model_name],
                        train_labels_device,
                        test_features[model_name],
                        test_labels_device,
                    )
                    bench_results[model_name] = metrics

                    tqdm.tqdm.write(
                        f"  {model_name}: acc={metrics['accuracy']:.2f}% "
                        f"bal_acc={metrics['balanced_accuracy']:.2f}% "
                        f"auc={metrics['auc']:.4f}"
                    )

                results[name] = bench_results

            # Sync all processes before next benchmark
            if self.distributed:
                dist.barrier()

        return results

    def get_primary_results(
        self, results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Extract results for the primary benchmark (used for early stopping).

        Args:
            results: Full results dict from tune()

        Returns:
            Dict with 'student' and 'teacher' metrics, or None if no primary benchmark
        """
        if self.primary_benchmark is None:
            return None
        return results.get(self.primary_benchmark)
