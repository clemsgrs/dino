import torch.nn as nn
from typing import List


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial training.

    MLP that predicts domain/center from feature embeddings.

    Args:
        input_dim: Dimension of input features (embed_dim from ViT).
        num_domains: Number of domains/centers to classify.
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        hidden_dims: List[int] = [1024, 512],
        dropout: float = 0.5,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_domains))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
