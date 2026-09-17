from torch import nn

from src.models.mlp import MLP
from src.models.cnn import CNN1D
from src.models.lstm import LSTMModel


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "MLP",
    "CNN1D",
    "LSTMModel",
    "count_parameters",
]