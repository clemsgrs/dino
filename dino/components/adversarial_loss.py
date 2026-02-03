import math
import torch.nn as nn
import torch.nn.functional as F


class DomainAdversarialLoss(nn.Module):
    """Domain adversarial loss with lambda scheduling.

    Computes cross-entropy loss for domain classification with a scheduled
    lambda parameter that controls the strength of gradient reversal.

    Lambda schedule follows DANN paper: λ = max_λ * (2 / (1 + exp(-γ*p)) - 1)
    where p is the training progress (0 to 1).

    Args:
        max_lambda: Maximum value for lambda (reached at end of training).
        gamma: Steepness of the lambda schedule curve.
        total_iterations: Total number of training iterations.
        warmup_pct: Fraction of training before adversarial loss kicks in.
    """

    def __init__(
        self,
        max_lambda: float = 1.0,
        gamma: float = 10.0,
        total_iterations: int = None,
        warmup_pct: float = 0.0,
    ):
        super().__init__()
        self.max_lambda = max_lambda
        self.gamma = gamma
        self.total_iterations = total_iterations
        self.warmup_iterations = (
            int(warmup_pct * total_iterations) if total_iterations else 0
        )

    def get_lambda(self, iteration: int) -> float:
        """Compute lambda value for current iteration."""
        if self.total_iterations is None:
            return self.max_lambda

        if iteration < self.warmup_iterations:
            return 0.0

        effective_iterations = self.total_iterations - self.warmup_iterations
        if effective_iterations <= 0:
            return self.max_lambda

        p = (iteration - self.warmup_iterations) / effective_iterations
        p = min(p, 1.0)
        return self.max_lambda * (2 / (1 + math.exp(-self.gamma * p)) - 1)

    def forward(self, domain_logits, domain_labels, iteration: int):
        """Compute domain classification loss.

        Args:
            domain_logits: Predicted domain logits [batch_size, num_domains].
            domain_labels: Ground truth domain indices [batch_size].
            iteration: Current training iteration.

        Returns:
            Tuple of (loss, lambda_value).
        """
        lambda_ = self.get_lambda(iteration)
        loss = F.cross_entropy(domain_logits, domain_labels)
        return loss, lambda_
