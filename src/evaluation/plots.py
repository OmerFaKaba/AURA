from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_loss_curves(
    history: Mapping[str, Sequence[float]],
    output_path: str | Path,
) -> Path:
    """Save training and validation MSE curves."""

    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)

    if len(train_loss) == 0 or len(train_loss) != len(val_loss):
        raise ValueError("Loss histories must be non-empty and equally sized.")

    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = int(np.argmin(val_loss)) + 1

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_loss, label="Training MSE", linewidth=2)
    axis.plot(epochs, val_loss, label="Validation MSE", linewidth=2)
    axis.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        alpha=0.6,
        label=f"Best epoch ({best_epoch})",
    )
    axis.set_title("LSTM training and validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss (normalized scale)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    path = _prepare_output_path(output_path)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_prediction_series(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot actual and predicted activity for sparse and dense test cells."""

    required = {"timestamp", "cell_id", "actual", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    activity = predictions.groupby("cell_id")["actual"].mean().sort_values()
    if activity.empty:
        raise ValueError("Prediction table cannot be empty.")

    selected_cells = [activity.index[0]]
    if len(activity) > 1 and activity.index[-1] != activity.index[0]:
        selected_cells.append(activity.index[-1])

    figure, axes = plt.subplots(
        len(selected_cells),
        1,
        figsize=(11, 4 * len(selected_cells)),
        sharex=True,
        squeeze=False,
    )

    for axis, cell_id in zip(axes[:, 0], selected_cells):
        cell_data = predictions.loc[
            predictions["cell_id"] == cell_id
        ].sort_values("timestamp")
        axis.plot(
            cell_data["timestamp"],
            cell_data["actual"],
            label="Actual",
            linewidth=2,
        )
        axis.plot(
            cell_data["timestamp"],
            cell_data["prediction"],
            label="Prediction",
            linewidth=2,
        )
        axis.set_title(f"Cell {cell_id}")
        axis.set_ylabel("Internet activity")
        axis.grid(alpha=0.25)
        axis.legend()

    axes[-1, 0].set_xlabel("Test timestamp")
    figure.autofmt_xdate()
    figure.suptitle("LSTM test predictions versus actual values", y=1.01)
    figure.tight_layout()

    path = _prepare_output_path(output_path)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_per_cell_mae(
    cell_mae: pd.Series,
    output_path: str | Path,
) -> Path:
    """Save the distribution of cell-level MAE values."""

    values = pd.Series(cell_mae, dtype=float).dropna()
    if values.empty:
        raise ValueError("Cell-level MAE values cannot be empty.")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values.to_numpy(), bins=30, edgecolor="black", alpha=0.8)
    axis.axvline(
        values.mean(),
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {values.mean():.2f}",
    )
    axis.axvline(
        values.median(),
        color="tab:green",
        linestyle=":",
        linewidth=2,
        label=f"Median: {values.median():.2f}",
    )
    axis.set_title("Distribution of test MAE across cells")
    axis.set_xlabel("Cell MAE")
    axis.set_ylabel("Number of cells")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()

    path = _prepare_output_path(output_path)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
