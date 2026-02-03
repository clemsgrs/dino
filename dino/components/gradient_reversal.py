import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer from DANN (Domain Adversarial Neural Networks).

    During forward pass, acts as identity. During backward pass, negates gradients
    and scales by lambda.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(torch.nn.Module):
    """Module wrapper for gradient reversal.

    Args:
        lambda_: Scaling factor for reversed gradients. Higher = stronger reversal.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)
