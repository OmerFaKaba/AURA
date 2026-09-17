import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (
    BASELINE_MODEL_PATH,
    BASELINE_SCALER_PATH,
    BASELINE_SEED,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    FIGURES_DIR,
    FORECAST_HORIZON,
    HOURLY_PATH,
    LEARNING_RATE,
    LSTM_HIDDEN_SIZE,
    MAX_EPOCHS,
    MODEL_OUTPUT_DIR,
    RESULTS_DIR,
    SELECTED_MODEL,
    TRAIN_RATIO,
    VAL_RATIO,
    WINDOW_SIZE,
)
from src.evaluation import (
    mae,
    mape,
    per_cell_mae,
    plot_loss_curves,
    plot_per_cell_mae,
    plot_prediction_series,
    rmse,
)
from src.models import LSTMModel, count_parameters
from src.training.normalize import apply_scaler, fit_scaler, inverse_scaler
from src.training.trainer import make_dataloader, predict_model, train_model
from src.training.windowing import make_windows, split_temporal, to_tensors


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the selected centralized LSTM baseline."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small sample and two epochs for a fast integration check.",
    )
    return parser.parse_args()


def configure_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_prediction_table(
    timestamps: np.ndarray,
    cell_ids: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    targets_flat = np.asarray(targets).reshape(-1)
    predictions_flat = np.asarray(predictions).reshape(-1)
    cell_ids_flat = np.asarray(cell_ids).reshape(-1)

    if not (
        len(timestamps)
        == len(cell_ids_flat)
        == len(targets_flat)
        == len(predictions_flat)
    ):
        raise ValueError("Prediction table inputs must have equal lengths.")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cell_id": cell_ids_flat,
            "actual": targets_flat,
            "prediction": predictions_flat,
            "absolute_error": np.abs(targets_flat - predictions_flat),
        }
    )


def main() -> None:
    args = parse_arguments()
    configure_reproducibility(BASELINE_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 2 if args.smoke else MAX_EPOCHS
    patience = 2 if args.smoke else EARLY_STOPPING_PATIENCE

    print(f"Device: {device}")
    print(f"Model: {SELECTED_MODEL}")
    print(f"Seed: {BASELINE_SEED}")
    print(f"Loading: {HOURLY_PATH}")

    df = pd.read_parquet(HOURLY_PATH)
    train_df, val_df, test_df = split_temporal(
        df,
        train=TRAIN_RATIO,
        val=VAL_RATIO,
    )

    stats = fit_scaler(train_df)
    train_scaled = apply_scaler(train_df, stats)
    val_scaled = apply_scaler(val_df, stats)
    test_scaled = apply_scaler(test_df, stats)

    X_train, y_train, _ = make_windows(
        train_scaled,
        window=WINDOW_SIZE,
        horizon=FORECAST_HORIZON,
    )
    X_val, y_val, _ = make_windows(
        val_scaled,
        window=WINDOW_SIZE,
        horizon=FORECAST_HORIZON,
    )
    X_test, y_test, test_cell_ids = make_windows(
        test_scaled,
        window=WINDOW_SIZE,
        horizon=FORECAST_HORIZON,
    )

    target_index = test_scaled.index[
        WINDOW_SIZE + FORECAST_HORIZON - 1 :
    ]
    test_timestamps = np.tile(target_index.to_numpy(), test_scaled.shape[1])

    X_train_tensor, y_train_tensor = to_tensors(X_train, y_train)
    X_val_tensor, y_val_tensor = to_tensors(X_val, y_val)
    X_test_tensor, y_test_tensor = to_tensors(X_test, y_test)

    if args.smoke:
        X_train_tensor = X_train_tensor[:2048]
        y_train_tensor = y_train_tensor[:2048]
        X_val_tensor = X_val_tensor[:512]
        y_val_tensor = y_val_tensor[:512]
        X_test_tensor = X_test_tensor[:512]
        y_test_tensor = y_test_tensor[:512]
        test_cell_ids = test_cell_ids[:512]
        test_timestamps = test_timestamps[:512]

    train_loader = make_dataloader(
        X_train_tensor,
        y_train_tensor,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=BASELINE_SEED,
    )
    val_loader = make_dataloader(
        X_val_tensor,
        y_val_tensor,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=BASELINE_SEED,
    )
    test_loader = make_dataloader(
        X_test_tensor,
        y_test_tensor,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=BASELINE_SEED,
    )

    model = LSTMModel(
        window_size=WINDOW_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
    )

    start_time = perf_counter()
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=LEARNING_RATE,
        patience=patience,
        seed=BASELINE_SEED,
        device=device,
    )
    training_seconds = perf_counter() - start_time

    predictions_scaled, targets_scaled = predict_model(
        model,
        test_loader,
        device=device,
    )
    predictions_real = inverse_scaler(
        predictions_scaled,
        stats,
        test_cell_ids,
    )
    targets_real = inverse_scaler(
        targets_scaled,
        stats,
        test_cell_ids,
    )

    metrics = {
        "model": SELECTED_MODEL,
        "seed": BASELINE_SEED,
        "test_mae": mae(targets_real, predictions_real),
        "test_rmse": rmse(targets_real, predictions_real),
        "test_mape": mape(targets_real, predictions_real),
        "parameters": count_parameters(model),
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "best_val_loss": float(min(history["val_loss"])),
        "training_seconds": training_seconds,
        "device": device,
    }

    prediction_table = build_prediction_table(
        test_timestamps,
        test_cell_ids,
        targets_real,
        predictions_real,
    )
    cell_mae = per_cell_mae(
        targets_real,
        predictions_real,
        test_cell_ids,
    )

    if args.smoke:
        assert len(prediction_table) == 512
        assert np.isfinite(list(metrics.values())[2:5]).all()
        print(json.dumps(metrics, indent=2))
        print("Baseline smoke validation passed.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history_path = RESULTS_DIR / "baseline_loss_history.csv"
    metrics_path = RESULTS_DIR / "baseline_metrics.json"
    predictions_path = RESULTS_DIR / "baseline_test_predictions.parquet"
    cell_mae_path = RESULTS_DIR / "baseline_per_cell_mae.csv"

    history_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
        }
    )
    history_df.to_csv(history_path, index=False)
    prediction_table.to_parquet(predictions_path, index=False)
    cell_mae.rename_axis("cell_id").reset_index().to_csv(
        cell_mae_path,
        index=False,
    )
    stats.to_parquet(BASELINE_SCALER_PATH)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    cpu_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
    }
    torch.save(
        {
            "model_name": SELECTED_MODEL,
            "state_dict": cpu_state,
            "window_size": WINDOW_SIZE,
            "hidden_size": LSTM_HIDDEN_SIZE,
            "seed": BASELINE_SEED,
            "metrics": metrics,
        },
        BASELINE_MODEL_PATH,
    )

    figure_paths = [
        plot_loss_curves(
            history,
            FIGURES_DIR / "baseline_loss_curve.png",
        ),
        plot_prediction_series(
            prediction_table,
            FIGURES_DIR / "baseline_prediction_vs_actual.png",
        ),
        plot_per_cell_mae(
            cell_mae,
            FIGURES_DIR / "baseline_per_cell_mae.png",
        ),
    ]

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {BASELINE_MODEL_PATH}")
    print(f"Saved scaler: {BASELINE_SCALER_PATH}")
    print(f"Saved metrics: {metrics_path}")
    for figure_path in figure_paths:
        print(f"Saved figure: {figure_path}")
    print("Baseline training completed.")


if __name__ == "__main__":
    main()
