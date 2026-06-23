"""Two-dimensional deterministic physics engine."""

from __future__ import annotations

from dataclasses import dataclass
import random

from config import SandboxConfig
from models.action import Action2D
from models.state import State2D
from utils.vectors import clamp_vector_components


@dataclass(frozen=True, slots=True)
class PhysicsResult2D:
    """Result of a 2D physics integration step."""

    requested_action: Action2D
    applied_action: Action2D
    next_state: State2D
    hit_boundary: tuple[bool, bool]


class PhysicsEngine2D:
    """Deterministic Newtonian dynamics for a single particle in 2D."""

    def __init__(self, config: SandboxConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng

    def reset(self) -> State2D:
        """Sample a fresh particle state from the configured ranges."""

        return State2D(
            position=(
                self.config.initial_position_x.sample(self.rng),
                self.config.initial_position_y.sample(self.rng),
            ),
            velocity=(
                self.config.initial_velocity_x.sample(self.rng),
                self.config.initial_velocity_y.sample(self.rng),
            ),
            acceleration=(0.0, 0.0),
            mass=self.config.mass_range.sample(self.rng),
        )

    def step(self, state: State2D, action: Action2D) -> PhysicsResult2D:
        """Advance the particle by one explicit-Euler integration step."""

        applied_force = clamp_vector_components(action.force, self.config.max_force)
        acceleration = (
            applied_force[0] / state.mass,
            applied_force[1] / state.mass,
        )
        position = (
            state.position[0] + state.velocity[0] * self.config.dt,
            state.position[1] + state.velocity[1] * self.config.dt,
        )
        velocity = (
            state.velocity[0] + acceleration[0] * self.config.dt,
            state.velocity[1] + acceleration[1] * self.config.dt,
        )

        min_x = self.config.world_bounds_x.minimum
        max_x = self.config.world_bounds_x.maximum
        min_y = self.config.world_bounds_y.minimum
        max_y = self.config.world_bounds_y.maximum

        hit_x = False
        hit_y = False

        if position[0] < min_x:
            position = (min_x, position[1])
            velocity = (0.0, velocity[1])
            hit_x = True
        elif position[0] > max_x:
            position = (max_x, position[1])
            velocity = (0.0, velocity[1])
            hit_x = True

        if position[1] < min_y:
            position = (position[0], min_y)
            velocity = (velocity[0], 0.0)
            hit_y = True
        elif position[1] > max_y:
            position = (position[0], max_y)
            velocity = (velocity[0], 0.0)
            hit_y = True

        next_state = State2D(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            mass=state.mass,
        )
        return PhysicsResult2D(
            requested_action=action,
            applied_action=Action2D(force=applied_force),
            next_state=next_state,
            hit_boundary=(hit_x, hit_y),
        )
