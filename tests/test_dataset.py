"""Dataset-building tests."""

from __future__ import annotations

import unittest

from config import SandboxConfig
from logging_tools.episode_logger import EpisodeLogger
from models.action import Action1D
from models.observation import Observation1D
from models.state import State1D
from models.transition import TransitionRecord
from training.dataset import HISTORY_LENGTH, INPUT_DIM, build_supervised_samples_1d


class DatasetTests(unittest.TestCase):
    """Validate history-window sample construction."""

    def test_sliding_window_count_and_shape(self) -> None:
        logger = EpisodeLogger()
        config = SandboxConfig(mode="1d", seed=3)
        initial_state = State1D(position=0.0, velocity=0.0, acceleration=0.0, mass=2.0)
        initial_observation = Observation1D(
            position=0.0,
            velocity=0.0,
            acceleration=0.0,
            previous_force=0.0,
            step_count=0,
        )
        episode = logger.start_episode(config, initial_state, initial_observation, policy_name="test")
        for step_index in range(10):
            observation_before = Observation1D(
                position=float(step_index),
                velocity=float(step_index) * 0.5,
                acceleration=0.0,
                previous_force=float(step_index - 1),
                step_count=step_index,
            )
            transition = TransitionRecord(
                episode_id=episode.episode_id,
                step_index=step_index + 1,
                timestamp_utc=logger.utc_now(),
                mode="1d",
                dt=0.1,
                state=State1D(
                    position=observation_before.position,
                    velocity=observation_before.velocity,
                    acceleration=0.0,
                    mass=2.0,
                ),
                observation_before=observation_before,
                requested_action=Action1D(force=1.0),
                applied_action=Action1D(force=1.0),
                next_state=State1D(
                    position=observation_before.position + 1.0,
                    velocity=observation_before.velocity + 0.1,
                    acceleration=0.5,
                    mass=2.0,
                ),
                observation_after=Observation1D(
                    position=observation_before.position + 1.0,
                    velocity=observation_before.velocity + 0.1,
                    acceleration=0.5,
                    previous_force=1.0,
                    step_count=step_index + 1,
                ),
                true_mass=2.0,
                hit_boundary=False,
                done=step_index == 9,
                done_reason="episode_limit" if step_index == 9 else None,
            )
            logger.record_transition(transition)

        finalized = logger.finalize_current_episode("episode_limit")
        features, targets, metadata = build_supervised_samples_1d([finalized], HISTORY_LENGTH)
        self.assertEqual(len(features), 3)
        self.assertEqual(metadata[0].step_index, 8)
        self.assertEqual(len(features[0]), INPUT_DIM)
        self.assertEqual(len(targets), 3)


if __name__ == "__main__":
    unittest.main()
