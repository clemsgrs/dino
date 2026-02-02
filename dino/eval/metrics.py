import numpy as np
from typing import Dict
from sklearn import metrics


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        labels: Ground truth labels of shape (N,)
        predictions: Predicted class labels of shape (N,)
        probabilities: Predicted probabilities of shape (N, num_classes)

    Returns:
        Dict with accuracy, balanced_accuracy, and auc
    """
    accuracy = metrics.accuracy_score(labels, predictions) * 100.0
    balanced_accuracy = metrics.balanced_accuracy_score(labels, predictions) * 100.0

    num_classes = probabilities.shape[1]
    if num_classes == 2:
        auc = metrics.roc_auc_score(labels, probabilities[:, 1])
    else:
        try:
            auc = metrics.roc_auc_score(labels, probabilities, multi_class="ovr")
        except ValueError:
            # Can happen if some classes are missing from test set
            auc = 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
    }
