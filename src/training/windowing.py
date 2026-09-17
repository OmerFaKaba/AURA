import torch
import numpy as np
import pandas as pd

def make_windows(
    df: pd.DataFrame,
    window: int = 24,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create input windows and targets separately for every cell."""

    if df.empty:
        raise ValueError("Cannot create windows from an empty DataFrame.")

    if window <= 0:
        raise ValueError("window must be greater than zero.")

    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    if len(df) < window + horizon:
        raise ValueError(
            f"At least {window + horizon} rows are required."
        )

    if df.isna().any().any():
        raise ValueError("DataFrame contains missing values.")

    samples_per_cell = len(df) - window - horizon + 1

    all_X = []
    all_y = []
    all_cell_ids = []

    for cell_id in df.columns:
        values = df[cell_id].to_numpy(dtype=np.float32)

        cell_windows = np.lib.stride_tricks.sliding_window_view(
            values,
            window_shape=window,
        )

        X_cell = cell_windows[:samples_per_cell]
        y_cell = values[
            window + horizon - 1:
            window + horizon - 1 + samples_per_cell
        ]

        cell_ids = np.repeat(cell_id, samples_per_cell)

        all_X.append(X_cell)
        all_y.append(y_cell)
        all_cell_ids.append(cell_ids)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    cell_ids = np.concatenate(all_cell_ids, axis=0)

    return X, y, cell_ids

def to_tensors(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert window arrays to the common PyTorch input format."""

    if X.ndim != 2:
        raise ValueError(
            f"X must have shape (samples, window), but received {X.shape}."
        )

    if y.ndim != 1:
        raise ValueError(
            f"y must have shape (samples,), but received {y.shape}."
        )

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
    ).unsqueeze(1)

    y_tensor = torch.as_tensor(
        y,
        dtype=torch.float32,
    ).unsqueeze(1)

    return X_tensor, y_tensor

def split_temporal(
    df: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame without shuffling."""

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if not 0 < train < 1:
        raise ValueError("train must be between 0 and 1.")

    if not 0 < val < 1:
        raise ValueError("val must be between 0 and 1.")

    if train + val >= 1:
        raise ValueError("train + val must be smaller than 1.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be chronologically ordered.")

    if not df.index.is_unique:
        raise ValueError("DataFrame index must contain unique timestamps.")

    n_rows = len(df)

    train_end = int(n_rows * train)
    val_end = int(n_rows * (train + val))

    if train_end == 0 or val_end == train_end or val_end == n_rows:
        raise ValueError("The dataset is too small for the requested split.")

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df