import torch
from torch import nn


class MLP(nn.Module):
    """Simple multilayer perceptron baseline."""

    def __init__(
        self,
        window_size: int = 24,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.window_size = window_size

        self.network = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, 1, {self.window_size}), "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[1] != 1 or x.shape[2] != self.window_size:
            raise ValueError(
                f"Expected input shape (batch, 1, {self.window_size}), "
                f"but received {tuple(x.shape)}."
            )

        x = x.squeeze(1)

        return self.network(x)