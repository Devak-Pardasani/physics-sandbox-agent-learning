"""Evaluation for the 1D dynamics model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from logging_tools.episode_io import load_episodes_from_directory
from models.action import Action1D
from models.observation import Observation1D
from models.state import State1D
from training.checkpoint import load_checkpoint
from training.dataset import HISTORY_LENGTH, build_history_window_1d, build_supervised_samples_1d


def evaluate_model(
    checkpoint_path: Path,
    test_dir: Path,
    artifact_dir: Path,
    max_rollout_steps: int = 50,
) -> dict[str, Any]:
    """Evaluate a trained dynamics model on held-out episodes."""

    bundle = load_checkpoint(checkpoint_path, device="cpu")
    episodes = load_episodes_from_directory(test_dir)
    features, targets, metadata = build_supervised_samples_1d(episodes, HISTORY_LENGTH)
    if not features:
        raise ValueError("Test directory must contain enough data for evaluation.")

    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    mass_errors: list[float] = []

    for feature_vector, target, sample_metadata in zip(features, targets, metadata):
        prediction = bundle.predict_acceleration(feature_vector)
        error = prediction - target
        absolute_errors.append(abs(error))
        squared_errors.append(error * error)
        applied_force = feature_vector[-1]
        if abs(applied_force) > 1e-6 and abs(prediction) > 1e-6:
            mass_estimate = applied_force / prediction
            mass_errors.append(abs(mass_estimate - sample_metadata.true_mass))

    rollout_velocity_errors: list[float] = []
    rollout_position_errors: list[float] = []
    for episode in episodes:
        rollout_metrics = _evaluate_rollout(bundle, episode, max_rollout_steps=max_rollout_steps)
        rollout_velocity_errors.extend(rollout_metrics["velocity_errors"])
        rollout_position_errors.extend(rollout_metrics["position_errors"])

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "test_dir": str(test_dir),
        "mode": "1d",
        "one_step_acceleration_mae": sum(absolute_errors) / len(absolute_errors),
        "one_step_acceleration_mse": sum(squared_errors) / len(squared_errors),
        "rollout_velocity_mae": (
            sum(rollout_velocity_errors) / len(rollout_velocity_errors)
            if rollout_velocity_errors
            else 0.0
        ),
        "rollout_position_mae": (
            sum(rollout_position_errors) / len(rollout_position_errors)
            if rollout_position_errors
            else 0.0
        ),
        "derived_mass_mae": sum(mass_errors) / len(mass_errors) if mass_errors else None,
        "evaluated_samples": len(features),
        "evaluated_episodes": len(episodes),
        "max_rollout_steps": max_rollout_steps,
    }

    eval_dir = artifact_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = eval_dir / f"evaluation_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


def _evaluate_rollout(bundle: Any, episode: Any, max_rollout_steps: int) -> dict[str, list[float]]:
    transitions = episode.transitions
    if len(transitions) < HISTORY_LENGTH:
        return {"velocity_errors": [], "position_errors": []}

    velocity_errors: list[float] = []
    position_errors: list[float] = []
    rollout_limit = min(len(transitions), max_rollout_steps)
    bounds = episode.config.get("world_bounds_x", {"minimum": float("-inf"), "maximum": float("inf")})
    min_x = float(bounds["minimum"])
    max_x = float(bounds["maximum"])

    current_state = transitions[HISTORY_LENGTH - 1].state
    history_entries = [
        {
            "position": transition.observation_before.position,
            "velocity": transition.observation_before.velocity,
            "previous_force": transition.observation_before.previous_force,
            "applied_force": transition.applied_action.force,
        }
        for transition in transitions[:HISTORY_LENGTH]
    ]

    for transition_index in range(HISTORY_LENGTH - 1, rollout_limit):
        feature_vector = []
        for entry in history_entries[-HISTORY_LENGTH:]:
            feature_vector.extend(
                [
                    float(entry["position"]),
                    float(entry["velocity"]),
                    float(entry["previous_force"]),
                    float(entry["applied_force"]),
                ]
            )

        predicted_acceleration = bundle.predict_acceleration(feature_vector)
        transition = transitions[transition_index]
        assert isinstance(current_state, State1D)
        assert isinstance(transition.applied_action, Action1D)
        predicted_position = current_state.position + current_state.velocity * transition.dt
        predicted_velocity = current_state.velocity + predicted_acceleration * transition.dt
        predicted_position, predicted_velocity = _apply_clamp_stop_1d(
            predicted_position,
            predicted_velocity,
            min_x=min_x,
            max_x=max_x,
        )
        predicted_next_state = State1D(
            position=predicted_position,
            velocity=predicted_velocity,
            acceleration=predicted_acceleration,
            mass=current_state.mass,
        )

        true_next_state = transition.next_state
        velocity_errors.append(abs(predicted_next_state.velocity - true_next_state.velocity))
        position_errors.append(abs(predicted_next_state.position - true_next_state.position))

        current_state = predicted_next_state
        if transition_index + 1 < len(transitions):
            next_transition = transitions[transition_index + 1]
            next_observation_before = Observation1D(
                position=predicted_next_state.position,
                velocity=predicted_next_state.velocity,
                acceleration=predicted_next_state.acceleration,
                previous_force=transition.applied_action.force,
                step_count=transition.step_index,
            )
            history_entries.append(
                {
                    "position": next_observation_before.position,
                    "velocity": next_observation_before.velocity,
                    "previous_force": next_observation_before.previous_force,
                    "applied_force": next_transition.applied_action.force,
                }
            )

    return {"velocity_errors": velocity_errors, "position_errors": position_errors}


def _apply_clamp_stop_1d(position: float, velocity: float, min_x: float, max_x: float) -> tuple[float, float]:
    if position < min_x:
        return min_x, 0.0
    if position > max_x:
        return max_x, 0.0
    return position, velocity
