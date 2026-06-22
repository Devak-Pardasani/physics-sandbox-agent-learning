"""Checkpoint loading and saving for the dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from training.model import DynamicsMLP


@dataclass(slots=True)
class InferenceBundle:
    """Loaded model and normalization stats for inference."""

    model: DynamicsMLP
    input_mean: torch.Tensor
    input_std: torch.Tensor
    target_mean: float
    target_std: float
    history_length: int
    feature_names: list[str]
    device: torch.device

    def predict_acceleration(self, features: list[float]) -> float:
        """Predict acceleration from one flattened feature vector."""

        tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized = (tensor - self.input_mean) / self.input_std
        self.model.eval()
        with torch.no_grad():
            output = self.model(normalized)
        acceleration = output.item() * self.target_std + self.target_mean
        return float(acceleration)


def save_checkpoint(
    path: Path,
    model: DynamicsMLP,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: float,
    target_std: float,
    history_length: int,
    feature_names: list[str],
) -> None:
    """Persist model weights and normalization metadata."""

    payload = {
        "model_state_dict": model.state_dict(),
        "input_mean": input_mean.cpu(),
        "input_std": input_std.cpu(),
        "target_mean": target_mean,
        "target_std": target_std,
        "history_length": history_length,
        "feature_names": feature_names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str | None = None) -> InferenceBundle:
    """Load a saved checkpoint for evaluation or replay overlay."""

    resolved_device = torch.device(device or "cpu")
    payload = torch.load(path, map_location=resolved_device)
    model = DynamicsMLP(input_dim=int(payload["input_mean"].numel()))
    model.load_state_dict(payload["model_state_dict"])
    model.to(resolved_device)
    return InferenceBundle(
        model=model,
        input_mean=payload["input_mean"].to(resolved_device),
        input_std=payload["input_std"].to(resolved_device),
        target_mean=float(payload["target_mean"]),
        target_std=float(payload["target_std"]),
        history_length=int(payload["history_length"]),
        feature_names=list(payload["feature_names"]),
        device=resolved_device,
    )
