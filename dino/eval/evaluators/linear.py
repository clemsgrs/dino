import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from .base import Evaluator
from ..metrics import compute_metrics


class LinearEvaluator(Evaluator):
    """Linear probe evaluator for feature evaluation.

    Trains a single linear layer on frozen features using SGD with cosine LR schedule.
    """

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 256,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
    ):
        """
        Args:
            epochs: Number of training epochs
            lr: Initial learning rate
            batch_size: Training batch size
            weight_decay: L2 regularization
            momentum: SGD momentum
        """
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum

    def evaluate(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Train linear classifier and compute metrics."""
        device = train_features.device
        embed_dim = train_features.shape[1]
        num_classes = len(torch.unique(train_labels))

        # Create linear classifier
        classifier = nn.Linear(embed_dim, num_classes).to(device)
        nn.init.xavier_uniform_(classifier.weight)
        nn.init.zeros_(classifier.bias)

        # Setup optimizer and scheduler
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.CrossEntropyLoss()

        # Create dataloader for training
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Training loop
        classifier.train()
        for _ in range(self.epochs):
            for features, labels in train_loader:
                optimizer.zero_grad()
                logits = classifier(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluation
        classifier.eval()
        with torch.no_grad():
            logits = classifier(test_features)
            probs = torch.softmax(logits, dim=1)
            _, predictions = logits.max(dim=1)

        probs_np = probs.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()

        return compute_metrics(test_labels_np, preds_np, probs_np)
