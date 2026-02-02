from abc import ABC, abstractmethod
from typing import Dict

import torch


class Evaluator(ABC):
    """Abstract base class for downstream evaluators."""

    @abstractmethod
    def evaluate(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate features on a downstream classification task.

        Args:
            train_features: Training set features of shape (N_train, embed_dim)
            train_labels: Training set labels of shape (N_train,)
            test_features: Test set features of shape (N_test, embed_dim)
            test_labels: Test set labels of shape (N_test,)

        Returns:
            Dict containing metrics: accuracy, balanced_accuracy, auc
        """
        pass
