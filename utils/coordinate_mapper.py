"""Helpers for mapping world-space coordinates into screen-space pixels."""

from __future__ import annotations

from dataclasses import dataclass


def map_scalar(
    value: float,
    source_range: tuple[float, float],
    target_range: tuple[float, float],
) -> float:
    """Map a scalar from one interval into another."""

    source_min, source_max = source_range
    target_min, target_max = target_range
    if source_max == source_min:
        return float(target_min)
    alpha = (value - source_min) / (source_max - source_min)
    return target_min + alpha * (target_max - target_min)


@dataclass(frozen=True, slots=True)
class CoordinateMapper:
    """Map world coordinates into a rectangular screen region."""

    world_x: tuple[float, float]
    world_y: tuple[float, float]
    screen_left: int
    screen_top: int
    screen_width: int
    screen_height: int

    @property
    def screen_right(self) -> int:
        return self.screen_left + self.screen_width

    @property
    def screen_bottom(self) -> int:
        return self.screen_top + self.screen_height

    def map_x(self, value: float) -> int:
        """Map an x coordinate into screen pixels."""

        return int(
            round(
                map_scalar(
                    value,
                    self.world_x,
                    (self.screen_left, self.screen_right),
                )
            )
        )

    def map_y(self, value: float) -> int:
        """Map a y coordinate into screen pixels with upward-positive world space."""

        return int(
            round(
                map_scalar(
                    value,
                    self.world_y,
                    (self.screen_bottom, self.screen_top),
                )
            )
        )

    def map_point(self, point: tuple[float, float]) -> tuple[int, int]:
        """Map a 2D point into screen pixels."""

        return self.map_x(point[0]), self.map_y(point[1])
