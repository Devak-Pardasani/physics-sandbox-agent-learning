"""Trajectory collection for the 1D learning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.base_policy import BasePolicy
from env.sandbox_env import SandboxEnv
from logging_tools.episode_logger import EpisodeLogger
from logging_tools.export_service import ExportService
from models.transition import TransitionRecord
from config import SandboxConfig


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    """Summary of a collection run."""

    episodes_collected: int
    output_dir: Path
    exported_paths: list[Path]


class TrajectoryCollector:
    """Collect per-episode transition logs from the environment."""

    def __init__(self, config: SandboxConfig, policy: BasePolicy, output_dir: Path) -> None:
        self.config = config
        self.policy = policy
        self.output_dir = output_dir
        self.export_service = ExportService(output_dir)

    def collect(self, episodes: int) -> CollectionSummary:
        """Collect the requested number of episodes and export them to disk."""

        if self.config.mode != "1d":
            raise ValueError("Trajectory collection currently supports mode='1d' only.")

        exported_paths: list[Path] = []
        env = SandboxEnv(self.config)
        logger = EpisodeLogger()
        try:
            for _ in range(episodes):
                env.reset()
                logger.start_episode(
                    config=self.config,
                    initial_state=env.current_state,
                    initial_observation=env.get_observation_model(),
                    policy_name=self.policy.name,
                )

                while not env.is_done:
                    observation_before = env.get_observation_model()
                    state_before = env.current_state
                    action = self.policy.act(observation_before, env.step_count)
                    _, _, done, info = env.step(action)
                    transition = TransitionRecord(
                        episode_id=logger.current_episode_id or "untracked",
                        step_index=env.step_count,
                        timestamp_utc=logger.utc_now(),
                        mode=self.config.mode,
                        dt=self.config.dt,
                        state=state_before,
                        observation_before=observation_before,
                        requested_action=env.current_requested_action,
                        applied_action=env.current_applied_action,
                        next_state=env.current_state,
                        observation_after=env.get_observation_model(),
                        true_mass=env.current_state.mass,
                        hit_boundary=info["hit_boundary"],
                        done=done,
                        done_reason=info["done_reason"],
                    )
                    logger.record_transition(transition)

                finalized = logger.finalize_current_episode("episode_limit")
                if finalized is None:
                    raise RuntimeError("Collector finalized an episode without a record.")
                exported_paths.append(self.export_service.export_episode(finalized))
        finally:
            env.close()

        return CollectionSummary(
            episodes_collected=episodes,
            output_dir=self.output_dir,
            exported_paths=exported_paths,
        )
