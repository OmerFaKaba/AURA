import numpy as np
import pandas as pd


def _prepare_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert metric inputs to validated one-dimensional arrays."""

    true_array = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    pred_array = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if len(true_array) == 0:
        raise ValueError("Metric inputs cannot be empty.")

    if len(true_array) != len(pred_array):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if not np.isfinite(true_array).all():
        raise ValueError("y_true contains non-finite values.")

    if not np.isfinite(pred_array).all():
        raise ValueError("y_pred contains non-finite values.")

    return true_array, pred_array


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate mean absolute error."""

    true_array, pred_array = _prepare_arrays(
        y_true,
        y_pred,
    )

    return float(
        np.mean(np.abs(true_array - pred_array))
    )


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate root mean squared error."""

    true_array, pred_array = _prepare_arrays(
        y_true,
        y_pred,
    )

    squared_error = (true_array - pred_array) ** 2

    return float(np.sqrt(np.mean(squared_error)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Calculate MAPE while excluding near-zero true values."""

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")

    true_array, pred_array = _prepare_arrays(
        y_true,
        y_pred,
    )

    valid_mask = np.abs(true_array) > epsilon

    if not valid_mask.any():
        raise ValueError(
            "MAPE cannot be calculated because all true values "
            "are zero or near zero."
        )

    percentage_errors = np.abs(
        (
            true_array[valid_mask]
            - pred_array[valid_mask]
        )
        / true_array[valid_mask]
    )

    return float(np.mean(percentage_errors) * 100)


def per_cell_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cell_ids: np.ndarray,
) -> pd.Series:
    """Calculate a separate MAE value for every cell."""

    true_array, pred_array = _prepare_arrays(
        y_true,
        y_pred,
    )

    cell_array = np.asarray(cell_ids).reshape(-1)

    if len(cell_array) != len(true_array):
        raise ValueError(
            "cell_ids must have the same length as y_true."
        )

    result = pd.DataFrame(
        {
            "cell_id": cell_array,
            "absolute_error": np.abs(
                true_array - pred_array
            ),
        }
    )

    cell_mae = result.groupby(
        "cell_id",
        sort=False,
    )["absolute_error"].mean()

    cell_mae.name = "mae"

    return cell_mae