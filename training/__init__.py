"""Training pipeline helpers for the physics sandbox."""

from .collector import CollectionSummary, TrajectoryCollector
from .dataset import HISTORY_LENGTH, SampleMetadata, build_supervised_samples_1d

__all__ = [
    "CollectionSummary",
    "HISTORY_LENGTH",
    "SampleMetadata",
    "TrajectoryCollector",
    "build_supervised_samples_1d",
]
