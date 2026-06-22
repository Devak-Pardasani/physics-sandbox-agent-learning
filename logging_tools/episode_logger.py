"""In-memory episode logging for the sandbox."""

from __future__ import annotations

from datetime import datetime, timezone

from config import SandboxConfig
from models.observation import ObservationModel
from models.state import StateModel
from models.transition import EpisodeRecord, TransitionRecord


class EpisodeLogger:
    """Store the active episode and retain the last completed episode."""

    def __init__(self) -> None:
        self.current_episode: EpisodeRecord | None = None
        self.last_completed_episode: EpisodeRecord | None = None
        self._episode_counter = 0

    @property
    def current_episode_id(self) -> str | None:
        """Return the current episode identifier."""

        if self.current_episode is None:
            return None
        return self.current_episode.episode_id

    def start_episode(
        self,
        config: SandboxConfig,
        initial_state: StateModel,
        initial_observation: ObservationModel,
        policy_name: str = "unknown",
    ) -> EpisodeRecord:
        """Begin logging a new episode."""

        self._episode_counter += 1
        started_at = self._utc_now()
        compact_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        episode_id = f"episode_{self._episode_counter:04d}_{compact_timestamp}"
        episode = EpisodeRecord(
            episode_id=episode_id,
            policy_name=policy_name,
            mode=config.mode,
            seed=config.seed,
            dt=config.dt,
            mass_range=config.mass_range.to_dict(),
            bounds_behavior=config.bounds_behavior,
            started_at_utc=started_at,
            config=config.to_dict(),
            initial_state=initial_state,
            initial_observation=initial_observation,
        )
        self.current_episode = episode
        return episode

    def record_transition(self, transition: TransitionRecord) -> None:
        """Append a transition to the active episode."""

        if self.current_episode is None:
            raise RuntimeError("Cannot record a transition without an active episode.")
        self.current_episode.transitions.append(transition)

    def finalize_current_episode(self, termination_reason: str | None) -> EpisodeRecord | None:
        """Close the active episode and optionally promote it to last completed."""

        if self.current_episode is None:
            return None

        finalized = self.current_episode.copy()
        finalized.ended_at_utc = self._utc_now()
        finalized.termination_reason = termination_reason
        if finalized.transitions:
            self.last_completed_episode = finalized
        self.current_episode = None
        return finalized

    def snapshot_current_episode(self, termination_reason: str = "snapshot") -> EpisodeRecord | None:
        """Return a detached snapshot of the active episode without finalizing it."""

        if self.current_episode is None:
            return None
        snapshot = self.current_episode.copy()
        snapshot.ended_at_utc = self._utc_now()
        snapshot.termination_reason = termination_reason
        return snapshot

    def get_exportable_episode(self) -> EpisodeRecord | None:
        """Return the best available episode for export."""

        if self.last_completed_episode is not None:
            return self.last_completed_episode.copy()
        return self.snapshot_current_episode()

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def utc_now(self) -> str:
        """Return the current UTC timestamp in ISO format."""

        return self._utc_now()
