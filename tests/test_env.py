"""Environment contract tests."""

from __future__ import annotations

import unittest

from config import SandboxConfig
from env.sandbox_env import SandboxEnv


class SandboxEnvTests(unittest.TestCase):
    """Validate the reusable RL-style environment API."""

    def test_observation_hides_mass_by_default(self) -> None:
        env = SandboxEnv(SandboxConfig(mode="1d", seed=1))
        observation = env.reset()
        self.assertNotIn("mass", observation)
        env.close()

    def test_observation_can_expose_mass(self) -> None:
        env = SandboxEnv(SandboxConfig(mode="2d", expose_mass_in_observation=True, seed=2))
        observation = env.reset()
        self.assertIn("mass", observation)
        env.close()

    def test_done_at_episode_limit(self) -> None:
        env = SandboxEnv(SandboxConfig(mode="1d", episode_length_limit=1, seed=3))
        _, _, done, info = env.step(0.0)
        self.assertTrue(done)
        self.assertEqual(info["done_reason"], "episode_limit")
        env.close()

    def test_info_contains_requested_and_applied_force(self) -> None:
        env = SandboxEnv(SandboxConfig(mode="2d", max_force=5.0, seed=4))
        _, _, _, info = env.step((10.0, -10.0))
        self.assertEqual(info["requested_force"], [10.0, -10.0])
        self.assertEqual(info["applied_force"], [5.0, -5.0])
        env.close()


if __name__ == "__main__":
    unittest.main()
