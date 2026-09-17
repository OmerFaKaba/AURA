import torch
from torch import nn


class LSTMModel(nn.Module):
    """LSTM baseline using the final hidden state."""

    def __init__(
        self,
        window_size: int = 24,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.window_size = window_size

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
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

        # (batch, 1, 24) -> (batch, 24, 1)
        x = x.transpose(1, 2)

        _, (hidden, _) = self.lstm(x)

        final_hidden = hidden[-1]

        return self.output(final_hidden)