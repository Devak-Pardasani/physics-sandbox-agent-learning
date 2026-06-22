"""Observation models for the physics sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from utils.vectors import Vector2, vector_to_list


@dataclass(frozen=True, slots=True)
class Observation1D:
    """Observation returned from the 1D environment."""

    position: float
    velocity: float
    acceleration: float
    previous_force: float
    step_count: int
    mass: float | None = None

    @property
    def mode(self) -> Literal["1d"]:
        return "1d"

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-friendly representation of the observation."""

        data: dict[str, float | int | str] = {
            "mode": self.mode,
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "previous_force": self.previous_force,
            "step_count": self.step_count,
        }
        if self.mass is not None:
            data["mass"] = self.mass
        return data


@dataclass(frozen=True, slots=True)
class Observation2D:
    """Observation returned from the 2D environment."""

    position: Vector2
    velocity: Vector2
    acceleration: Vector2
    previous_force: Vector2
    step_count: int
    mass: float | None = None

    @property
    def mode(self) -> Literal["2d"]:
        return "2d"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the observation."""

        data: dict[str, object] = {
            "mode": self.mode,
            "position": vector_to_list(self.position),
            "velocity": vector_to_list(self.velocity),
            "acceleration": vector_to_list(self.acceleration),
            "previous_force": vector_to_list(self.previous_force),
            "step_count": self.step_count,
        }
        if self.mass is not None:
            data["mass"] = self.mass
        return data


ObservationModel = Observation1D | Observation2D
