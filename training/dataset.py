"""Dataset building utilities for the 1D dynamics learner."""

from __future__ import annotations

from dataclasses import dataclass

from models.action import Action1D
from models.observation import Observation1D
from models.transition import EpisodeRecord, TransitionRecord

HISTORY_LENGTH = 8
FEATURES_PER_STEP = 4
INPUT_DIM = HISTORY_LENGTH * FEATURES_PER_STEP


@dataclass(frozen=True, slots=True)
class SampleMetadata:
    """Metadata associated with a single supervised training sample."""

    episode_id: str
    step_index: int
    true_mass: float


def build_supervised_samples_1d(
    episodes: list[EpisodeRecord],
    history_length: int = HISTORY_LENGTH,
) -> tuple[list[list[float]], list[float], list[SampleMetadata]]:
    """Build fixed-window 1D supervised samples from logged episodes."""

    features: list[list[float]] = []
    targets: list[float] = []
    metadata: list[SampleMetadata] = []

    for episode in episodes:
        if episode.mode != "1d":
            raise ValueError("1D supervised sampling only supports episodes with mode='1d'.")
        for end_index in range(history_length - 1, len(episode.transitions)):
            transition = episode.transitions[end_index]
            features.append(build_history_window_1d(episode.transitions, end_index, history_length))
            targets.append(transition.next_state.acceleration)
            metadata.append(
                SampleMetadata(
                    episode_id=episode.episode_id,
                    step_index=transition.step_index,
                    true_mass=transition.true_mass,
                )
            )
    return features, targets, metadata


def build_history_window_1d(
    transitions: list[TransitionRecord],
    end_index: int,
    history_length: int = HISTORY_LENGTH,
) -> list[float]:
    """Flatten a fixed history window ending at the specified transition index."""

    window = transitions[end_index - history_length + 1 : end_index + 1]
    if len(window) != history_length:
        raise ValueError("History window does not have the requested length.")

    flattened: list[float] = []
    for transition in window:
        flattened.extend(build_transition_features_1d(transition))
    return flattened


def build_transition_features_1d(transition: TransitionRecord) -> list[float]:
    """Build per-transition input features for the 1D model."""

    if transition.mode != "1d":
        raise ValueError("Transition feature extraction only supports 1D transitions.")
    observation = transition.observation_before
    action = transition.applied_action
    if not isinstance(observation, Observation1D) or not isinstance(action, Action1D):
        raise TypeError("Expected a 1D observation and 1D action.")
    return [
        observation.position,
        observation.velocity,
        observation.previous_force,
        action.force,
    ]


def feature_names(history_length: int = HISTORY_LENGTH) -> list[str]:
    """Return the flattened feature names used by the 1D model."""

    names: list[str] = []
    for offset in range(history_length):
        prefix = f"t-{history_length - 1 - offset}"
        names.extend(
            [
                f"{prefix}_position",
                f"{prefix}_velocity",
                f"{prefix}_previous_force",
                f"{prefix}_applied_force",
            ]
        )
    return names
