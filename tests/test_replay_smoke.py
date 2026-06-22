"""Headless replay smoke tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from agents.random_policy import RandomUniformPolicy
from config import SandboxConfig
from logging_tools.episode_io import load_episodes_from_directory
from training.collector import TrajectoryCollector

PYGAME_AVAILABLE = importlib.util.find_spec("pygame") is not None


class ReplaySmokeTests(unittest.TestCase):
    """Ensure replay can initialize and render headlessly."""

    @unittest.skipUnless(PYGAME_AVAILABLE, "Pygame is required for replay smoke tests.")
    def test_replay_window_initializes(self) -> None:
        from app_controller import ReplayController

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = SandboxConfig(mode="1d", seed=5, episode_length_limit=10)
            policy = RandomUniformPolicy(max_force=config.max_force, seed=5)
            collector = TrajectoryCollector(config=config, policy=policy, output_dir=output_dir)
            collector.collect(episodes=1)
            episode = load_episodes_from_directory(output_dir)[0]

            replay_config = SandboxConfig.from_dict({**episode.config, "mode": episode.mode})
            controller = ReplayController(episode=episode, config=replay_config)
            controller.render_frame()
            controller.tick(events=[])
            controller.close()


if __name__ == "__main__":
    unittest.main()
