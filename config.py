"""Configuration objects for the physics sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import random
from typing import Any, Literal

Mode = Literal["1d", "2d"]
BoundsBehavior = Literal["clamp_stop"]


@dataclass(frozen=True, slots=True)
class NumericRange:
    """Inclusive numeric range used for sampling and validation."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError(
                f"Invalid range: minimum {self.minimum} is greater than maximum {self.maximum}."
            )

    def sample(self, rng: random.Random) -> float:
        """Sample a deterministic value from the range using the provided RNG."""

        return rng.uniform(self.minimum, self.maximum)

    def contains_range(self, other: "NumericRange") -> bool:
        """Return True when the other range is fully inside this range."""

        return self.minimum <= other.minimum and other.maximum <= self.maximum

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly representation of the range."""

        return {"minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NumericRange":
        """Rebuild a range from a serialized snapshot."""

        return cls(minimum=float(payload["minimum"]), maximum=float(payload["maximum"]))


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Top-level configuration for the physics sandbox desktop application."""

    mode: Mode = "1d"
    screen_width: int = 1180
    screen_height: int = 720
    window_title: str = "Physics Sandbox"
    dt: float = 1.0 / 60.0
    target_fps: int = 60
    mass_range: NumericRange = field(default_factory=lambda: NumericRange(1.0, 5.0))
    max_force: float = 60.0
    force_adjust_step: float = 5.0
    world_bounds_x: NumericRange = field(
        default_factory=lambda: NumericRange(-300.0, 300.0)
    )
    world_bounds_y: NumericRange = field(
        default_factory=lambda: NumericRange(-200.0, 200.0)
    )
    initial_position_x: NumericRange = field(
        default_factory=lambda: NumericRange(-150.0, 150.0)
    )
    initial_position_y: NumericRange = field(
        default_factory=lambda: NumericRange(-100.0, 100.0)
    )
    initial_velocity_x: NumericRange = field(
        default_factory=lambda: NumericRange(-40.0, 40.0)
    )
    initial_velocity_y: NumericRange = field(
        default_factory=lambda: NumericRange(-40.0, 40.0)
    )
    show_mass: bool = False
    show_debug: bool = True
    show_force_vector: bool = True
    expose_mass_in_observation: bool = False
    episode_length_limit: int = 1200
    object_radius: int = 16
    bounds_behavior: BoundsBehavior = "clamp_stop"
    export_dir: Path = field(default_factory=lambda: Path("exports"))
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("1d", "2d"):
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("Screen dimensions must be positive.")
        if self.dt <= 0.0:
            raise ValueError("Timestep dt must be positive.")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive.")
        if self.max_force < 0.0:
            raise ValueError("max_force must be non-negative.")
        if self.force_adjust_step <= 0.0:
            raise ValueError("force_adjust_step must be positive.")
        if self.episode_length_limit <= 0:
            raise ValueError("episode_length_limit must be positive.")
        if self.object_radius <= 0:
            raise ValueError("object_radius must be positive.")
        if self.bounds_behavior != "clamp_stop":
            raise ValueError(f"Unsupported bounds behavior: {self.bounds_behavior}")
        if not self.world_bounds_x.contains_range(self.initial_position_x):
            raise ValueError("initial_position_x must fit inside world_bounds_x.")
        if not self.world_bounds_y.contains_range(self.initial_position_y):
            raise ValueError("initial_position_y must fit inside world_bounds_y.")

    def with_mode(self, mode: Mode) -> "SandboxConfig":
        """Return a copy of the config with a different simulation mode."""

        return replace(self, mode=mode)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly configuration snapshot."""

        return {
            "mode": self.mode,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "window_title": self.window_title,
            "dt": self.dt,
            "target_fps": self.target_fps,
            "mass_range": self.mass_range.to_dict(),
            "max_force": self.max_force,
            "force_adjust_step": self.force_adjust_step,
            "world_bounds_x": self.world_bounds_x.to_dict(),
            "world_bounds_y": self.world_bounds_y.to_dict(),
            "initial_position_x": self.initial_position_x.to_dict(),
            "initial_position_y": self.initial_position_y.to_dict(),
            "initial_velocity_x": self.initial_velocity_x.to_dict(),
            "initial_velocity_y": self.initial_velocity_y.to_dict(),
            "show_mass": self.show_mass,
            "show_debug": self.show_debug,
            "show_force_vector": self.show_force_vector,
            "expose_mass_in_observation": self.expose_mass_in_observation,
            "episode_length_limit": self.episode_length_limit,
            "object_radius": self.object_radius,
            "bounds_behavior": self.bounds_behavior,
            "export_dir": str(self.export_dir),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SandboxConfig":
        """Rebuild a config from a serialized snapshot."""

        return cls(
            mode=payload.get("mode", "1d"),
            screen_width=int(payload.get("screen_width", 1180)),
            screen_height=int(payload.get("screen_height", 720)),
            window_title=payload.get("window_title", "Physics Sandbox"),
            dt=float(payload.get("dt", 1.0 / 60.0)),
            target_fps=int(payload.get("target_fps", 60)),
            mass_range=NumericRange.from_dict(payload.get("mass_range", {"minimum": 1.0, "maximum": 5.0})),
            max_force=float(payload.get("max_force", 60.0)),
            force_adjust_step=float(payload.get("force_adjust_step", 5.0)),
            world_bounds_x=NumericRange.from_dict(
                payload.get("world_bounds_x", {"minimum": -300.0, "maximum": 300.0})
            ),
            world_bounds_y=NumericRange.from_dict(
                payload.get("world_bounds_y", {"minimum": -200.0, "maximum": 200.0})
            ),
            initial_position_x=NumericRange.from_dict(
                payload.get("initial_position_x", {"minimum": -150.0, "maximum": 150.0})
            ),
            initial_position_y=NumericRange.from_dict(
                payload.get("initial_position_y", {"minimum": -100.0, "maximum": 100.0})
            ),
            initial_velocity_x=NumericRange.from_dict(
                payload.get("initial_velocity_x", {"minimum": -40.0, "maximum": 40.0})
            ),
            initial_velocity_y=NumericRange.from_dict(
                payload.get("initial_velocity_y", {"minimum": -40.0, "maximum": 40.0})
            ),
            show_mass=bool(payload.get("show_mass", False)),
            show_debug=bool(payload.get("show_debug", True)),
            show_force_vector=bool(payload.get("show_force_vector", True)),
            expose_mass_in_observation=bool(payload.get("expose_mass_in_observation", False)),
            episode_length_limit=int(payload.get("episode_length_limit", 1200)),
            object_radius=int(payload.get("object_radius", 16)),
            bounds_behavior=payload.get("bounds_behavior", "clamp_stop"),
            export_dir=Path(payload.get("export_dir", "exports")),
            seed=payload.get("seed"),
        )
