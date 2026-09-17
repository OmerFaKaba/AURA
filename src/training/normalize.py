import numpy as np
import pandas as pd


def fit_scaler(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cell-level normalization statistics from training data only."""

    if train_df.empty:
        raise ValueError("Cannot fit scaler on an empty DataFrame.")

    means = train_df.mean(axis=0)
    stds = train_df.std(axis=0, ddof=0)

    # Constant cells cannot be divided by zero.
    stds = stds.replace(0, 1.0)

    stats = pd.DataFrame(
        {
            "mean": means,
            "std": stds,
        }
    )

    return stats


def apply_scaler(
    df: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """Apply previously calculated statistics to a DataFrame."""

    missing_cells = df.columns.difference(stats.index)

    if not missing_cells.empty:
        raise ValueError(
            f"Scaler statistics are missing for cells: {missing_cells.tolist()}"
        )

    means = stats.loc[df.columns, "mean"]
    stds = stats.loc[df.columns, "std"]

    scaled_df = (df - means) / stds

    return scaled_df


def inverse_scaler(
    values: np.ndarray,
    stats: pd.DataFrame,
    cell_ids: np.ndarray,
) -> np.ndarray:
    """Convert normalized predictions back to their original scale."""

    values_array = np.asarray(values, dtype=float)
    original_shape = values_array.shape
    flat_values = values_array.reshape(-1)

    cell_ids_array = np.asarray(cell_ids).reshape(-1)

    if len(flat_values) != len(cell_ids_array):
        raise ValueError("values and cell_ids must have the same length.")

    missing_cells = pd.Index(cell_ids_array).difference(stats.index)

    if not missing_cells.empty:
        raise ValueError(
            f"Scaler statistics are missing for cells: {missing_cells.tolist()}"
        )

    means = stats.loc[cell_ids_array, "mean"].to_numpy()
    stds = stats.loc[cell_ids_array, "std"].to_numpy()

    restored = flat_values * stds + means

    return restored.reshape(original_shape)