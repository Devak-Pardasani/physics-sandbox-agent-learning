"""Episode serialization and migration-aware loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.action import Action1D, Action2D, ActionModel
from models.observation import Observation1D, Observation2D, ObservationModel
from models.state import State1D, State2D, StateModel
from models.transition import EpisodeRecord, TransitionRecord


def load_episode(path: Path) -> EpisodeRecord:
    """Load a single episode JSON file, accepting both current and legacy schemas."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    mode = payload["mode"]
    initial_state = _parse_state(
        mode,
        payload["initial_state"],
        mass=payload.get("true_mass"),
    )
    initial_observation = _parse_observation(mode, payload["initial_observation"])

    transitions_payload = payload.get("transitions", [])
    previous_force = 0.0 if mode == "1d" else (0.0, 0.0)
    transitions: list[TransitionRecord] = []

    for raw_transition in transitions_payload:
        requested_action = _parse_action(mode, raw_transition["requested_action"])
        applied_action = _parse_action(mode, raw_transition["applied_action"])
        state = _parse_state(mode, raw_transition["state"], mass=raw_transition["true_mass"])
        next_state = _parse_state(mode, raw_transition["next_state"], mass=raw_transition["true_mass"])

        if "observation_before" in raw_transition and "observation_after" in raw_transition:
            observation_before = _parse_observation(mode, raw_transition["observation_before"])
            observation_after = _parse_observation(mode, raw_transition["observation_after"])
        else:
            observation_before = _observation_from_state(
                mode=mode,
                state=state,
                previous_force=previous_force,
                step_count=max(0, raw_transition["step_index"] - 1),
            )
            observation_after = _parse_observation(mode, raw_transition["observation"])

        transition = TransitionRecord(
            episode_id=raw_transition["episode_id"],
            step_index=raw_transition["step_index"],
            timestamp_utc=raw_transition["timestamp_utc"],
            mode=mode,
            dt=raw_transition["dt"],
            state=state,
            observation_before=observation_before,
            requested_action=requested_action,
            applied_action=applied_action,
            next_state=next_state,
            observation_after=observation_after,
            true_mass=raw_transition["true_mass"],
            hit_boundary=_parse_hit_boundary(raw_transition["hit_boundary"]),
            done=raw_transition["done"],
            done_reason=raw_transition.get("done_reason"),
        )
        transitions.append(transition)
        previous_force = _force_from_action(applied_action)

    return EpisodeRecord(
        episode_id=payload["episode_id"],
        policy_name=payload.get("policy_name", "legacy"),
        mode=mode,
        seed=payload.get("seed", payload.get("config", {}).get("seed")),
        dt=payload.get("dt", payload.get("config", {}).get("dt")),
        mass_range=payload.get(
            "mass_range",
            payload.get("config", {}).get("mass_range", {"minimum": 0.0, "maximum": 0.0}),
        ),
        bounds_behavior=payload.get(
            "bounds_behavior",
            payload.get("config", {}).get("bounds_behavior", "clamp_stop"),
        ),
        started_at_utc=payload["started_at_utc"],
        config=payload.get("config", {}),
        initial_state=initial_state,
        initial_observation=initial_observation,
        transitions=transitions,
        ended_at_utc=payload.get("ended_at_utc"),
        termination_reason=payload.get("termination_reason"),
    )


def load_episodes_from_directory(directory: Path) -> list[EpisodeRecord]:
    """Load all episode JSON files from a directory in filename order."""

    return [load_episode(path) for path in sorted(directory.glob("*.json"))]


def _parse_state(mode: str, payload: dict[str, Any], mass: float | None) -> StateModel:
    if mode == "1d":
        return State1D(
            position=float(payload["position"]),
            velocity=float(payload["velocity"]),
            acceleration=float(payload["acceleration"]),
            mass=float(payload.get("mass", 0.0 if mass is None else mass)),
        )
    return State2D(
        position=(float(payload["position"][0]), float(payload["position"][1])),
        velocity=(float(payload["velocity"][0]), float(payload["velocity"][1])),
        acceleration=(float(payload["acceleration"][0]), float(payload["acceleration"][1])),
        mass=float(payload.get("mass", 0.0 if mass is None else mass)),
    )


def _parse_observation(mode: str, payload: dict[str, Any]) -> ObservationModel:
    if mode == "1d":
        return Observation1D(
            position=float(payload["position"]),
            velocity=float(payload["velocity"]),
            acceleration=float(payload["acceleration"]),
            previous_force=float(payload["previous_force"]),
            step_count=int(payload["step_count"]),
            mass=float(payload["mass"]) if "mass" in payload else None,
        )
    return Observation2D(
        position=(float(payload["position"][0]), float(payload["position"][1])),
        velocity=(float(payload["velocity"][0]), float(payload["velocity"][1])),
        acceleration=(float(payload["acceleration"][0]), float(payload["acceleration"][1])),
        previous_force=(float(payload["previous_force"][0]), float(payload["previous_force"][1])),
        step_count=int(payload["step_count"]),
        mass=float(payload["mass"]) if "mass" in payload else None,
    )


def _parse_action(mode: str, payload: dict[str, Any]) -> ActionModel:
    if mode == "1d":
        return Action1D(force=float(payload["force"]))
    return Action2D(force=(float(payload["force"][0]), float(payload["force"][1])))


def _parse_hit_boundary(value: Any) -> bool | tuple[bool, bool]:
    if isinstance(value, list):
        return bool(value[0]), bool(value[1])
    return bool(value)


def _observation_from_state(
    mode: str,
    state: StateModel,
    previous_force: float | tuple[float, float],
    step_count: int,
) -> ObservationModel:
    if mode == "1d":
        assert isinstance(state, State1D)
        assert isinstance(previous_force, float)
        return Observation1D(
            position=state.position,
            velocity=state.velocity,
            acceleration=state.acceleration,
            previous_force=previous_force,
            step_count=step_count,
        )
    assert isinstance(state, State2D)
    assert isinstance(previous_force, tuple)
    return Observation2D(
        position=state.position,
        velocity=state.velocity,
        acceleration=state.acceleration,
        previous_force=previous_force,
        step_count=step_count,
    )


def _force_from_action(action: ActionModel) -> float | tuple[float, float]:
    if isinstance(action, Action1D):
        return action.force
    return action.force
