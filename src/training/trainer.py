import math
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Create a reproducible PyTorch DataLoader."""

    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a PyTorch tensor.")

    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a PyTorch tensor.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    dataset = TensorDataset(X, y)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )

    return loader

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    min_delta: float = 0.0,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train a model and restore the weights with the best validation loss."""

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if lr <= 0:
        raise ValueError("lr must be greater than zero.")

    if patience <= 0:
        raise ValueError("patience must be greater than zero.")

    if min_delta < 0:
        raise ValueError("min_delta cannot be negative.")

    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset cannot be empty.")

    if len(val_loader.dataset) == 0:
        raise ValueError("Validation dataset cannot be empty.")

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()

        train_loss_sum = 0.0
        train_sample_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = X_batch.shape[0]

            train_loss_sum += loss.item() * batch_size
            train_sample_count += batch_size

        train_loss = train_loss_sum / train_sample_count

        model.eval()

        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                batch_size = X_batch.shape[0]

                val_loss_sum += loss.item() * batch_size
                val_sample_count += batch_size

        val_loss = val_loss_sum / val_sample_count

        if not math.isfinite(train_loss):
            raise RuntimeError("Training loss is not finite.")

        if not math.isfinite(val_loss):
            raise RuntimeError("Validation loss is not finite.")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    return model, history

def predict_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model predictions and targets in DataLoader order."""

    if len(data_loader.dataset) == 0:
        raise ValueError("Prediction dataset cannot be empty.")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)

    model.eval()

    prediction_batches = []
    target_batches = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            predictions = model(X_batch)

            prediction_batches.append(
                predictions.cpu().numpy()
            )

            target_batches.append(
                y_batch.cpu().numpy()
            )

    all_predictions = np.concatenate(
        prediction_batches,
        axis=0,
    )

    all_targets = np.concatenate(
        target_batches,
        axis=0,
    )

    return all_predictions, all_targets
