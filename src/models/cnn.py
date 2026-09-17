import torch
from torch import nn


class CNN1D(nn.Module):
    """One-dimensional convolutional baseline."""

    def __init__(
        self,
        window_size: int = 24,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        output_length = window_size - 2 * (kernel_size - 1)

        if output_length <= 0:
            raise ValueError(
                "window_size is too small for two convolution layers."
            )

        self.window_size = window_size

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=kernel_size,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=kernel_size,
            ),
            nn.ReLU(),
        )

        self.output = nn.Linear(
            64 * output_length,
            1,
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

        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        return self.output(x)