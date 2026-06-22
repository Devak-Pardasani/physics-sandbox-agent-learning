"""State models for the physics sandbox."""

from __future__ import annotations

from dataclasses import dataclass

from utils.vectors import Vector2, vector_to_list


@dataclass(frozen=True, slots=True)
class State1D:
    """State for a single particle moving along one axis."""

    position: float
    velocity: float
    acceleration: float
    mass: float

    def to_dict(self, include_mass: bool = True) -> dict[str, float]:
        """Return a JSON-friendly representation of the state."""

        data = {
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
        }
        if include_mass:
            data["mass"] = self.mass
        return data


@dataclass(frozen=True, slots=True)
class State2D:
    """State for a single particle moving in two dimensions."""

    position: Vector2
    velocity: Vector2
    acceleration: Vector2
    mass: float

    def to_dict(self, include_mass: bool = True) -> dict[str, object]:
        """Return a JSON-friendly representation of the state."""

        data: dict[str, object] = {
            "position": vector_to_list(self.position),
            "velocity": vector_to_list(self.velocity),
            "acceleration": vector_to_list(self.acceleration),
        }
        if include_mass:
            data["mass"] = self.mass
        return data


StateModel = State1D | State2D
