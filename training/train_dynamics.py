"""Training routine for the 1D acceleration predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from logging_tools.episode_io import load_episodes_from_directory
from training.checkpoint import save_checkpoint
from training.dataset import HISTORY_LENGTH, build_supervised_samples_1d, feature_names
from training.model import DynamicsMLP


def train_dynamics(
    train_dir: Path,
    val_dir: Path,
    artifact_dir: Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the 1D acceleration predictor and save the best checkpoint."""

    train_episodes = load_episodes_from_directory(train_dir)
    val_episodes = load_episodes_from_directory(val_dir)
    train_features, train_targets, _ = build_supervised_samples_1d(train_episodes, HISTORY_LENGTH)
    val_features, val_targets, _ = build_supervised_samples_1d(val_episodes, HISTORY_LENGTH)

    if not train_features or not val_features:
        raise ValueError("Training and validation directories must contain enough data for history windows.")

    if verbose:
        print(
            "Training dynamics model "
            f"(train_episodes={len(train_episodes)}, val_episodes={len(val_episodes)}, "
            f"train_samples={len(train_features)}, val_samples={len(val_features)}, "
            f"device={device})"
        )

    device_obj = torch.device(device)
    train_x = torch.tensor(train_features, dtype=torch.float32, device=device_obj)
    train_y = torch.tensor(train_targets, dtype=torch.float32, device=device_obj)
    val_x = torch.tensor(val_features, dtype=torch.float32, device=device_obj)
    val_y = torch.tensor(val_targets, dtype=torch.float32, device=device_obj)

    input_mean = train_x.mean(dim=0, keepdim=True)
    input_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    target_mean = float(train_y.mean().item())
    target_std = float(train_y.std().clamp_min(1e-6).item())

    norm_train_x = (train_x - input_mean) / input_std
    norm_val_x = (val_x - input_mean) / input_std
    norm_train_y = (train_y - target_mean) / target_std
    norm_val_y = (val_y - target_mean) / target_std

    dataset = TensorDataset(norm_train_x, norm_train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DynamicsMLP(input_dim=norm_train_x.shape[1]).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    model_dir = artifact_dir / "models"
    metrics_path = model_dir / "dynamics_1d_metrics.json"
    metadata_path = model_dir / "dynamics_1d_metadata.json"
    checkpoint_path = model_dir / "dynamics_1d.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.shape[0]
            sample_count += batch_x.shape[0]

        train_loss = running_loss / sample_count

        model.eval()
        with torch.no_grad():
            val_predictions = model(norm_val_x)
            val_loss = loss_fn(val_predictions, norm_val_y).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path,
                model=model,
                input_mean=input_mean,
                input_std=input_std,
                target_mean=target_mean,
                target_std=target_std,
                history_length=HISTORY_LENGTH,
                feature_names=feature_names(HISTORY_LENGTH),
            )

        if verbose:
            suffix = " | saved best checkpoint" if improved else ""
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"best_val_loss={best_val_loss:.6f}{suffix}"
            )

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    metadata = {
        "target": "acceleration",
        "mode": "1d",
        "history_length": HISTORY_LENGTH,
        "input_dim": norm_train_x.shape[1],
        "hidden_layers": [128, 128],
        "activation": "ReLU",
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "checkpoint_path": str(checkpoint_path),
        "feature_names": feature_names(HISTORY_LENGTH),
        "input_mean": input_mean.squeeze(0).cpu().tolist(),
        "input_std": input_std.squeeze(0).cpu().tolist(),
        "target_mean": target_mean,
        "target_std": target_std,
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    if verbose:
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Metrics: {metrics_path}")

    return {
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "metadata_path": metadata_path,
        "best_val_loss": best_val_loss,
    }
