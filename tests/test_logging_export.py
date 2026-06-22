"""Episode logging and export tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from config import SandboxConfig
from logging_tools.episode_io import load_episode
from logging_tools.episode_logger import EpisodeLogger
from logging_tools.export_service import ExportService
from models.action import Action1D
from models.observation import Observation1D
from models.state import State1D
from models.transition import TransitionRecord


class LoggingExportTests(unittest.TestCase):
    """Validate episode metadata and JSON export round-tripping."""

    def test_export_round_trip_preserves_new_schema(self) -> None:
        logger = EpisodeLogger()
        config = SandboxConfig(mode="1d", seed=7, dt=0.2)
        initial_state = State1D(position=1.0, velocity=2.0, acceleration=0.0, mass=3.5)
        initial_observation = Observation1D(
            position=1.0,
            velocity=2.0,
            acceleration=0.0,
            previous_force=0.0,
            step_count=0,
        )
        episode = logger.start_episode(
            config=config,
            initial_state=initial_state,
            initial_observation=initial_observation,
            policy_name="random_uniform",
        )
        logger.record_transition(
            TransitionRecord(
                episode_id=episode.episode_id,
                step_index=1,
                timestamp_utc=logger.utc_now(),
                mode="1d",
                dt=config.dt,
                state=initial_state,
                observation_before=initial_observation,
                requested_action=Action1D(force=12.0),
                applied_action=Action1D(force=12.0),
                next_state=State1D(position=1.4, velocity=2.8, acceleration=4.0, mass=3.5),
                observation_after=Observation1D(
                    position=1.4,
                    velocity=2.8,
                    acceleration=4.0,
                    previous_force=12.0,
                    step_count=1,
                ),
                true_mass=3.5,
                hit_boundary=False,
                done=True,
                done_reason="episode_limit",
            )
        )
        finalized = logger.finalize_current_episode("episode_limit")
        self.assertIsNotNone(finalized)
        assert finalized is not None

        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = ExportService(Path(temp_dir)).export_episode(finalized)
            loaded = load_episode(export_path)

        self.assertEqual(loaded.policy_name, "random_uniform")
        self.assertEqual(loaded.dt, 0.2)
        self.assertEqual(loaded.mass_range, config.mass_range.to_dict())
        self.assertEqual(loaded.bounds_behavior, "clamp_stop")
        self.assertEqual(loaded.initial_state.mass, 3.5)
        self.assertEqual(loaded.transitions[0].observation_before.previous_force, 0.0)
        self.assertEqual(loaded.transitions[0].observation_after.previous_force, 12.0)
        self.assertEqual(loaded.transitions[0].true_mass, 3.5)


if __name__ == "__main__":
    unittest.main()
