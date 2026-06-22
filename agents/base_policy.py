"""Base policy interfaces for environment interaction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.action import ActionModel
from models.observation import ObservationModel


class BasePolicy(ABC):
    """Minimal policy interface for acting inside the sandbox."""

    name: str

    @abstractmethod
    def act(self, observation: ObservationModel, step_index: int) -> ActionModel:
        """Return an action for the current observation."""
