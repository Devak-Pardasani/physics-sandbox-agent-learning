"""Action models for the physics sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from utils.vectors import Vector2, vector_to_list


@dataclass(frozen=True, slots=True)
class Action1D:
    """Force action for the 1D environment."""

    force: float

    @property
    def mode(self) -> Literal["1d"]:
        return "1d"

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-friendly representation of the action."""

        return {"mode": self.mode, "force": self.force}


@dataclass(frozen=True, slots=True)
class Action2D:
    """Force action for the 2D environment."""

    force: Vector2

    @property
    def mode(self) -> Literal["2d"]:
        return "2d"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the action."""

        return {"mode": self.mode, "force": vector_to_list(self.force)}


ActionModel = Action1D | Action2D
