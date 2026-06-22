"""Random exploration policy used for initial data collection."""

from __future__ import annotations

import random

from agents.base_policy import BasePolicy
from models.action import Action1D, Action2D, ActionModel
from models.observation import Observation1D, Observation2D, ObservationModel


class RandomUniformPolicy(BasePolicy):
    """Sample actions uniformly within the configured force bounds."""

    name = "random_uniform"

    def __init__(self, max_force: float, seed: int | None = None) -> None:
        self.max_force = max_force
        self.rng = random.Random(seed)

    def act(self, observation: ObservationModel, step_index: int) -> ActionModel:
        if isinstance(observation, Observation1D):
            return Action1D(force=self.rng.uniform(-self.max_force, self.max_force))
        if isinstance(observation, Observation2D):
            return Action2D(
                force=(
                    self.rng.uniform(-self.max_force, self.max_force),
                    self.rng.uniform(-self.max_force, self.max_force),
                )
            )
        raise TypeError(f"Unsupported observation type: {type(observation)!r}")
