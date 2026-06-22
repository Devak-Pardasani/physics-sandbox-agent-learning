"""Evaluation helpers for trained sandbox models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_model(
    checkpoint_path: Path,
    test_dir: Path,
    artifact_dir: Path,
    max_rollout_steps: int = 50,
) -> dict[str, Any]:
    """Evaluate a trained model, importing PyTorch-backed code on demand."""

    from .evaluate_model import evaluate_model as _evaluate_model

    return _evaluate_model(
        checkpoint_path=checkpoint_path,
        test_dir=test_dir,
        artifact_dir=artifact_dir,
        max_rollout_steps=max_rollout_steps,
    )

__all__ = ["evaluate_model"]
