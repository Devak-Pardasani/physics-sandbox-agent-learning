"""Collector and random-policy tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agents.random_policy import RandomUniformPolicy
from config import SandboxConfig
from logging_tools.episode_io import load_episodes_from_directory
from models.observation import Observation1D
from training.collector import TrajectoryCollector


class CollectorTests(unittest.TestCase):
    """Validate random action generation and episode export."""

    def test_random_policy_stays_within_force_bounds(self) -> None:
        policy = RandomUniformPolicy(max_force=7.5, seed=1)
        observation = Observation1D(
            position=0.0,
            velocity=0.0,
            acceleration=0.0,
            previous_force=0.0,
            step_count=0,
        )
        for step_index in range(100):
            action = policy.act(observation, step_index)
            self.assertGreaterEqual(action.force, -7.5)
            self.assertLessEqual(action.force, 7.5)

    def test_collect_writes_episode_files_with_new_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = SandboxConfig(mode="1d", seed=2, episode_length_limit=10)
            policy = RandomUniformPolicy(max_force=config.max_force, seed=2)
            collector = TrajectoryCollector(config=config, policy=policy, output_dir=output_dir)

            summary = collector.collect(episodes=2)

            self.assertEqual(summary.episodes_collected, 2)
            episodes = load_episodes_from_directory(output_dir)
            self.assertEqual(len(episodes), 2)
            first_transition = episodes[0].transitions[0]
            self.assertIsNotNone(first_transition.observation_before)
            self.assertIsNotNone(first_transition.observation_after)


if __name__ == "__main__":
    unittest.main()
