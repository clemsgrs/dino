import torch
import numpy as np
from typing import Dict

from .base import Evaluator
from ..metrics import compute_metrics


class KNNEvaluator(Evaluator):
    """K-Nearest Neighbors classifier for feature evaluation.

    Uses cosine similarity via normalized dot product with temperature-weighted
    soft voting. Processes in chunks for memory efficiency.
    """

    def __init__(self, k: int = 20, temperature: float = 0.07):
        """
        Args:
            k: Number of nearest neighbors
            temperature: Temperature for soft voting weights
        """
        self.k = k
        self.temperature = temperature

    def evaluate(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Run KNN classification and compute metrics.

        Features should already be L2 normalized for cosine similarity.
        """
        device = train_features.device
        num_classes = len(torch.unique(train_labels))

        # Transpose train features for efficient matrix multiplication
        train_features_t = train_features.t()

        num_test = test_labels.shape[0]
        num_chunks = min(num_test, 100)
        chunk_size = num_test // num_chunks

        all_probs = []
        all_preds = []
        retrieval_one_hot = torch.zeros(self.k, num_classes, device=device)

        for idx in range(0, num_test, chunk_size):
            end_idx = min(idx + chunk_size, num_test)
            features = test_features[idx:end_idx]
            targets = test_labels[idx:end_idx]
            batch_size = targets.shape[0]

            # Compute cosine similarity via dot product (features assumed normalized)
            similarity = torch.mm(features, train_features_t)
            distances, indices = similarity.topk(self.k, largest=True, sorted=True)

            # Get labels of k nearest neighbors
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            # Build one-hot encoding of neighbor labels
            retrieval_one_hot.resize_(batch_size * self.k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            # Temperature-weighted voting
            distances_transform = distances.clone().div_(self.temperature).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                dim=1,
            )

            # Normalize probabilities
            probs = probs / probs.sum(dim=-1, keepdim=True)
            _, predictions = probs.max(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(predictions.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        test_labels_np = test_labels.cpu().numpy()

        return compute_metrics(test_labels_np, all_preds, all_probs)
