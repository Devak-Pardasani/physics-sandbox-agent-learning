"""Physics-engine tests."""

from __future__ import annotations

import random
import unittest

from config import NumericRange, SandboxConfig
from env.physics_1d import PhysicsEngine1D
from env.physics_2d import PhysicsEngine2D
from models.action import Action1D, Action2D
from models.state import State1D, State2D


class PhysicsEngineTests(unittest.TestCase):
    """Validate deterministic physics integration behavior."""

    def test_explicit_euler_1d(self) -> None:
        config = SandboxConfig(dt=0.5, max_force=10.0)
        engine = PhysicsEngine1D(config, random.Random(1))
        state = State1D(position=1.0, velocity=3.0, acceleration=0.0, mass=2.0)

        result = engine.step(state, Action1D(force=4.0))

        self.assertAlmostEqual(result.next_state.position, 2.5)
        self.assertAlmostEqual(result.next_state.velocity, 4.0)
        self.assertAlmostEqual(result.next_state.acceleration, 2.0)

    def test_force_is_clipped_to_max(self) -> None:
        config = SandboxConfig(dt=0.25, max_force=5.0)
        engine = PhysicsEngine1D(config, random.Random(2))
        state = State1D(position=0.0, velocity=0.0, acceleration=0.0, mass=1.0)

        result = engine.step(state, Action1D(force=100.0))

        self.assertEqual(result.applied_action.force, 5.0)
        self.assertAlmostEqual(result.next_state.acceleration, 5.0)

    def test_clamp_stop_bounds_in_2d(self) -> None:
        config = SandboxConfig(
            dt=1.0,
            max_force=50.0,
            world_bounds_x=NumericRange(-1.0, 1.0),
            world_bounds_y=NumericRange(-1.0, 1.0),
            initial_position_x=NumericRange(0.0, 0.0),
            initial_position_y=NumericRange(0.0, 0.0),
        )
        engine = PhysicsEngine2D(config, random.Random(3))
        state = State2D(
            position=(0.8, 0.0),
            velocity=(1.0, 0.25),
            acceleration=(0.0, 0.0),
            mass=1.0,
        )

        result = engine.step(state, Action2D(force=(0.0, 0.0)))

        self.assertEqual(result.next_state.position, (1.0, 0.25))
        self.assertEqual(result.next_state.velocity, (0.0, 0.25))
        self.assertEqual(result.hit_boundary, (True, False))


if __name__ == "__main__":
    unittest.main()
