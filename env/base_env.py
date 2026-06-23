"""Abstract base environment interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

Observation = dict[str, Any]
StepResult = tuple[Observation, float, bool, dict[str, Any]]


class BaseEnv(ABC):
    """Minimal RL-style environment interface."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """Advance the environment by one step."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Return the current observation without stepping."""

    @abstractmethod
    def render(self) -> None:
        """Render the environment to its current output target."""

    @abstractmethod
    def close(self) -> None:
        """Release resources owned by the environment."""
