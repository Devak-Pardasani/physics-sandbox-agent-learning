"""Shared vector and scalar helpers used across the app."""

from __future__ import annotations

from typing import TypeAlias

Vector2: TypeAlias = tuple[float, float]


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a scalar value to the closed interval [minimum, maximum]."""

    return max(minimum, min(value, maximum))


def clamp_vector_components(vector: Vector2, limit: float) -> Vector2:
    """Clamp each vector component independently to [-limit, limit]."""

    return (
        clamp(vector[0], -limit, limit),
        clamp(vector[1], -limit, limit),
    )


def vector_is_near_zero(vector: Vector2, tolerance: float = 1e-6) -> bool:
    """Return True when the vector magnitude is effectively zero."""

    return abs(vector[0]) <= tolerance and abs(vector[1]) <= tolerance


def vector_to_list(vector: Vector2) -> list[float]:
    """Convert a vector tuple to a JSON-friendly list."""

    return [float(vector[0]), float(vector[1])]
