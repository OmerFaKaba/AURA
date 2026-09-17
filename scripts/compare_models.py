import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (
    BATCH_SIZE,
    COMPARISON_SEEDS,
    EARLY_STOPPING_PATIENCE,
    HOURLY_PATH,
    LEARNING_RATE,
    MAX_EPOCHS,
    WINDOW_SIZE,
)
from src.evaluation import mae, rmse, mape
from src.models import MLP, CNN1D, LSTMModel, count_parameters
from src.training.normalize import (
    fit_scaler,
    apply_scaler,
    inverse_scaler,
)
from src.training.trainer import (
    make_dataloader,
    train_model,
    predict_model,
)
from src.training.windowing import (
    split_temporal,
    make_windows,
    to_tensors,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short integration test instead of the full experiment.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Device:", device)
    print("Loading:", HOURLY_PATH)

    # --------------------------------------------------
    # Load and split data
    # --------------------------------------------------

    df = pd.read_parquet(HOURLY_PATH)

    train_df, val_df, test_df = split_temporal(df)

    assert train_df.shape == (352, 400)
    assert val_df.shape == (76, 400)
    assert test_df.shape == (76, 400)

    # --------------------------------------------------
    # Fit scaler on training data only
    # --------------------------------------------------

    stats = fit_scaler(train_df)

    train_scaled = apply_scaler(train_df, stats)
    val_scaled = apply_scaler(val_df, stats)
    test_scaled = apply_scaler(test_df, stats)

    # --------------------------------------------------
    # Create windows separately
    # --------------------------------------------------

    X_train, y_train, train_cell_ids = make_windows(
        train_scaled,
        window=WINDOW_SIZE,
    )

    X_val, y_val, val_cell_ids = make_windows(
        val_scaled,
        window=WINDOW_SIZE,
    )

    X_test, y_test, test_cell_ids = make_windows(
        test_scaled,
        window=WINDOW_SIZE,
    )

    # --------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------

    X_train_tensor, y_train_tensor = to_tensors(
        X_train,
        y_train,
    )

    X_val_tensor, y_val_tensor = to_tensors(
        X_val,
        y_val,
    )

    X_test_tensor, y_test_tensor = to_tensors(
        X_test,
        y_test,
    )

    # --------------------------------------------------
    # Smoke or full configuration
    # --------------------------------------------------

    if args.smoke:
        seeds = [42]
        epochs = 2
        patience = 2

        X_train_tensor = X_train_tensor[:2048]
        y_train_tensor = y_train_tensor[:2048]

        X_val_tensor = X_val_tensor[:512]
        y_val_tensor = y_val_tensor[:512]
        val_cell_ids = val_cell_ids[:512]

        X_test_tensor = X_test_tensor[:512]
        y_test_tensor = y_test_tensor[:512]
        test_cell_ids = test_cell_ids[:512]

        print("Running smoke validation.")

    else:
        seeds = COMPARISON_SEEDS
        epochs = MAX_EPOCHS
        patience = EARLY_STOPPING_PATIENCE

        print("Running full comparison.")

    # --------------------------------------------------
    # Validation and test DataLoaders
    # --------------------------------------------------

    val_loader = make_dataloader(
        X_val_tensor,
        y_val_tensor,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=42,
    )

    test_loader = make_dataloader(
        X_test_tensor,
        y_test_tensor,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=42,
    )

    # --------------------------------------------------
    # Model factories
    # --------------------------------------------------

    model_factories = {
        "MLP": lambda: MLP(
            window_size=WINDOW_SIZE,
            hidden_size=64,
        ),
        "CNN1D": lambda: CNN1D(
            window_size=WINDOW_SIZE,
            kernel_size=5,
        ),
        "LSTM": lambda: LSTMModel(
            window_size=WINDOW_SIZE,
            hidden_size=64,
        ),
    }

    run_results = []
    history_results = []

    # --------------------------------------------------
    # Model comparison
    # --------------------------------------------------

    for model_name, model_factory in model_factories.items():
        for seed in seeds:
            print()
            print(
                "Starting",
                model_name,
                "| seed:",
                seed,
                "| maximum epochs:",
                epochs,
            )

            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            train_loader = make_dataloader(
                X_train_tensor,
                y_train_tensor,
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=seed,
            )

            model = model_factory()

            start_time = perf_counter()

            trained_model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=LEARNING_RATE,
                patience=patience,
                seed=seed,
                device=device,
            )

            training_seconds = perf_counter() - start_time

            # --------------------------------------------------
            # Store epoch-level loss history
            # --------------------------------------------------

            best_epoch = int(
                np.argmin(history["val_loss"])
            ) + 1

            best_val_loss = min(
                history["val_loss"]
            )

            for epoch_number, (
                train_loss,
                val_loss,
            ) in enumerate(
                zip(
                    history["train_loss"],
                    history["val_loss"],
                ),
                start=1,
            ):
                history_results.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "epoch": epoch_number,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )

            # --------------------------------------------------
            # Validation predictions
            # --------------------------------------------------

            val_predictions_scaled, val_targets_scaled = predict_model(
                trained_model,
                val_loader,
                device=device,
            )

            val_predictions_real = inverse_scaler(
                val_predictions_scaled,
                stats,
                val_cell_ids,
            )

            val_targets_real = inverse_scaler(
                val_targets_scaled,
                stats,
                val_cell_ids,
            )

            val_mae_result = mae(
                val_targets_real,
                val_predictions_real,
            )

            # --------------------------------------------------
            # Test predictions
            # --------------------------------------------------

            test_predictions_scaled, test_targets_scaled = predict_model(
                trained_model,
                test_loader,
                device=device,
            )

            test_predictions_real = inverse_scaler(
                test_predictions_scaled,
                stats,
                test_cell_ids,
            )

            test_targets_real = inverse_scaler(
                test_targets_scaled,
                stats,
                test_cell_ids,
            )

            test_mae_result = mae(
                test_targets_real,
                test_predictions_real,
            )

            test_rmse_result = rmse(
                test_targets_real,
                test_predictions_real,
            )

            test_mape_result = mape(
                test_targets_real,
                test_predictions_real,
            )

            # --------------------------------------------------
            # Store run results
            # --------------------------------------------------

            result = {
                "model": model_name,
                "seed": seed,
                "val_mae": val_mae_result,
                "test_mae": test_mae_result,
                "test_rmse": test_rmse_result,
                "test_mape": test_mape_result,
                "parameters": count_parameters(trained_model),
                "training_seconds": training_seconds,
                "epochs_completed": len(history["train_loss"]),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }

            run_results.append(result)

            print(
                model_name,
                "| seed:",
                seed,
                "| val MAE:",
                round(val_mae_result, 4),
                "| test MAE:",
                round(test_mae_result, 4),
                "| best epoch:",
                best_epoch,
                "| completed epochs:",
                len(history["train_loss"]),
                "| seconds:",
                round(training_seconds, 2),
            )

    # --------------------------------------------------
    # Convert results to DataFrames
    # --------------------------------------------------

    runs_df = pd.DataFrame(run_results)
    history_df = pd.DataFrame(history_results)

    expected_run_count = len(model_factories) * len(seeds)

    assert len(runs_df) == expected_run_count
    assert not history_df.empty

    assert np.isfinite(runs_df["val_mae"]).all()
    assert np.isfinite(runs_df["test_mae"]).all()
    assert np.isfinite(runs_df["test_rmse"]).all()
    assert np.isfinite(runs_df["test_mape"]).all()
    assert np.isfinite(runs_df["best_val_loss"]).all()
    assert (runs_df["training_seconds"] > 0).all()

    assert np.isfinite(history_df["train_loss"]).all()
    assert np.isfinite(history_df["val_loss"]).all()

    # --------------------------------------------------
    # Create comparison summary
    # --------------------------------------------------

    summary = runs_df.groupby(
        "model",
        as_index=False,
    ).agg(
        val_mae_mean=("val_mae", "mean"),
        val_mae_std=("val_mae", "std"),
        test_mae_mean=("test_mae", "mean"),
        test_mae_std=("test_mae", "std"),
        test_rmse_mean=("test_rmse", "mean"),
        test_mape_mean=("test_mape", "mean"),
        parameters=("parameters", "first"),
        training_seconds_mean=("training_seconds", "mean"),
        epochs_mean=("epochs_completed", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        best_val_loss_mean=("best_val_loss", "mean"),
    )

    summary["val_mae_std"] = summary[
        "val_mae_std"
    ].fillna(0.0)

    summary["test_mae_std"] = summary[
        "test_mae_std"
    ].fillna(0.0)

    summary = summary.sort_values(
        "val_mae_mean",
        ascending=True,
    ).reset_index(drop=True)

    selected_model = summary.iloc[0]["model"]

    # --------------------------------------------------
    # Display comparison
    # --------------------------------------------------

    print()
    print("Comparison:")
    print(summary.to_string(index=False))
    print()
    print("Selected architecture:", selected_model)

    # --------------------------------------------------
    # Validate smoke run or save full results
    # --------------------------------------------------

    if args.smoke:
        assert len(runs_df) == 3
        assert len(summary) == 3

        print("T2.9 smoke validation passed.")

    else:
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        runs_path = results_dir / "model_runs.csv"
        summary_path = results_dir / "model_comparison.csv"
        history_path = results_dir / "loss_history.csv"

        runs_df.to_csv(
            runs_path,
            index=False,
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        history_df.to_csv(
            history_path,
            index=False,
        )

        print("Saved:", runs_path)
        print("Saved:", summary_path)
        print("Saved:", history_path)
        print("T2.9 comparison completed.")


if __name__ == "__main__":
    main()
