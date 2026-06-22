"""Transition and episode logging models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import Mode
from models.action import ActionModel
from models.observation import ObservationModel
from models.state import StateModel


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One logged transition from the environment."""

    episode_id: str
    step_index: int
    timestamp_utc: str
    mode: Mode
    dt: float
    state: StateModel
    observation_before: ObservationModel
    requested_action: ActionModel
    applied_action: ActionModel
    next_state: StateModel
    observation_after: ObservationModel
    true_mass: float
    hit_boundary: bool | tuple[bool, bool]
    done: bool
    done_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the transition."""

        hit_boundary: bool | list[bool]
        if isinstance(self.hit_boundary, tuple):
            hit_boundary = [self.hit_boundary[0], self.hit_boundary[1]]
        else:
            hit_boundary = self.hit_boundary

        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "timestamp_utc": self.timestamp_utc,
            "mode": self.mode,
            "dt": self.dt,
            "state": self.state.to_dict(include_mass=False),
            "observation_before": self.observation_before.to_dict(),
            "requested_action": self.requested_action.to_dict(),
            "applied_action": self.applied_action.to_dict(),
            "next_state": self.next_state.to_dict(include_mass=False),
            "observation_after": self.observation_after.to_dict(),
            "true_mass": self.true_mass,
            "hit_boundary": hit_boundary,
            "done": self.done,
            "done_reason": self.done_reason,
        }


@dataclass(slots=True)
class EpisodeRecord:
    """In-memory record of a sandbox episode."""

    episode_id: str
    policy_name: str
    mode: Mode
    seed: int | None
    dt: float
    mass_range: dict[str, float]
    bounds_behavior: str
    started_at_utc: str
    config: dict[str, Any]
    initial_state: StateModel
    initial_observation: ObservationModel
    transitions: list[TransitionRecord] = field(default_factory=list)
    ended_at_utc: str | None = None
    termination_reason: str | None = None

    @property
    def step_count(self) -> int:
        """Return the number of logged transitions."""

        return len(self.transitions)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the episode."""

        return {
            "episode_id": self.episode_id,
            "policy_name": self.policy_name,
            "mode": self.mode,
            "seed": self.seed,
            "dt": self.dt,
            "mass_range": self.mass_range,
            "bounds_behavior": self.bounds_behavior,
            "true_mass": self.initial_state.mass,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "termination_reason": self.termination_reason,
            "step_count": self.step_count,
            "config": self.config,
            "initial_state": self.initial_state.to_dict(include_mass=False),
            "initial_observation": self.initial_observation.to_dict(),
            "transitions": [transition.to_dict() for transition in self.transitions],
        }

    def copy(self) -> "EpisodeRecord":
        """Return a detached copy that can be exported safely."""

        return EpisodeRecord(
            episode_id=self.episode_id,
            policy_name=self.policy_name,
            mode=self.mode,
            seed=self.seed,
            dt=self.dt,
            mass_range=dict(self.mass_range),
            bounds_behavior=self.bounds_behavior,
            started_at_utc=self.started_at_utc,
            config=dict(self.config),
            initial_state=self.initial_state,
            initial_observation=self.initial_observation,
            transitions=list(self.transitions),
            ended_at_utc=self.ended_at_utc,
            termination_reason=self.termination_reason,
        )
