"""One-dimensional deterministic physics engine."""

from __future__ import annotations

from dataclasses import dataclass
import random

from config import SandboxConfig
from models.action import Action1D
from models.state import State1D
from utils.vectors import clamp


@dataclass(frozen=True, slots=True)
class PhysicsResult1D:
    """Result of a 1D physics integration step."""

    requested_action: Action1D
    applied_action: Action1D
    next_state: State1D
    hit_boundary: bool


class PhysicsEngine1D:
    """Deterministic Newtonian dynamics for a single particle in 1D."""

    def __init__(self, config: SandboxConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng

    def reset(self) -> State1D:
        """Sample a fresh particle state from the configured ranges."""

        return State1D(
            position=self.config.initial_position_x.sample(self.rng),
            velocity=self.config.initial_velocity_x.sample(self.rng),
            acceleration=0.0,
            mass=self.config.mass_range.sample(self.rng),
        )

    def step(self, state: State1D, action: Action1D) -> PhysicsResult1D:
        """Advance the particle by one explicit-Euler integration step."""

        applied_force = clamp(action.force, -self.config.max_force, self.config.max_force)
        acceleration = applied_force / state.mass
        position = state.position + state.velocity * self.config.dt
        velocity = state.velocity + acceleration * self.config.dt

        min_x = self.config.world_bounds_x.minimum
        max_x = self.config.world_bounds_x.maximum
        hit_boundary = False

        if position < min_x:
            position = min_x
            velocity = 0.0
            hit_boundary = True
        elif position > max_x:
            position = max_x
            velocity = 0.0
            hit_boundary = True

        next_state = State1D(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            mass=state.mass,
        )
        return PhysicsResult1D(
            requested_action=action,
            applied_action=Action1D(force=applied_force),
            next_state=next_state,
            hit_boundary=hit_boundary,
        )
