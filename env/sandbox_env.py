"""Reusable RL-style environment wrapper for the physics sandbox."""

from __future__ import annotations

from collections.abc import Sequence
import random
from typing import TYPE_CHECKING, Any

from config import Mode, SandboxConfig
from env.base_env import BaseEnv, Observation, StepResult
from env.physics_1d import PhysicsEngine1D
from env.physics_2d import PhysicsEngine2D
from models.action import Action1D, Action2D, ActionModel
from models.observation import Observation1D, Observation2D, ObservationModel
from models.state import State1D, State2D, StateModel
from utils.vectors import Vector2

if TYPE_CHECKING:
    from ui.hud import HudData, HudRenderer
    from ui.renderer import SandboxRenderer


class SandboxEnv(BaseEnv):
    """Single-particle deterministic sandbox with a clean RL-style API."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._physics_1d = PhysicsEngine1D(config, self._rng) if config.mode == "1d" else None
        self._physics_2d = PhysicsEngine2D(config, self._rng) if config.mode == "2d" else None
        self._state: StateModel | None = None
        self._requested_action: ActionModel = self._zero_action(config.mode)
        self._applied_action: ActionModel = self._zero_action(config.mode)
        self._step_count = 0
        self._done = False
        self._renderer: SandboxRenderer | None = None
        self._hud: HudRenderer | None = None
        self.reset()

    @property
    def mode(self) -> Mode:
        """Return the active simulation mode."""

        return self.config.mode

    @property
    def step_count(self) -> int:
        """Return the number of completed environment steps."""

        return self._step_count

    @property
    def is_done(self) -> bool:
        """Return whether the episode is currently terminated."""

        return self._done

    @property
    def current_state(self) -> StateModel:
        """Return the current internal particle state."""

        return self._require_state()

    @property
    def current_requested_action(self) -> ActionModel:
        """Return the most recently requested action."""

        return self._requested_action

    @property
    def current_applied_action(self) -> ActionModel:
        """Return the most recently applied action after clipping."""

        return self._applied_action

    def reset(self) -> Observation:
        """Reset physics state and return the starting observation."""

        self._step_count = 0
        self._done = False
        self._requested_action = self._zero_action(self.mode)
        self._applied_action = self._zero_action(self.mode)

        if self.mode == "1d":
            assert self._physics_1d is not None
            self._state = self._physics_1d.reset()
        else:
            assert self._physics_2d is not None
            self._state = self._physics_2d.reset()

        return self.get_observation()

    def step(self, action: Any) -> StepResult:
        """Advance the environment by one step using the provided action."""

        if self._done:
            return self.get_observation(), 0.0, True, self._build_info(hit_boundary=self._empty_boundary())

        if self.mode == "1d":
            assert self._physics_1d is not None
            state = self._require_state_1d()
            requested_action = self._coerce_action_1d(action)
            result = self._physics_1d.step(state, requested_action)
        else:
            assert self._physics_2d is not None
            state = self._require_state_2d()
            requested_action = self._coerce_action_2d(action)
            result = self._physics_2d.step(state, requested_action)

        self._state = result.next_state
        self._requested_action = result.requested_action
        self._applied_action = result.applied_action
        self._step_count += 1
        self._done = self._step_count >= self.config.episode_length_limit

        observation = self.get_observation()
        info = self._build_info(hit_boundary=result.hit_boundary)
        return observation, 0.0, self._done, info

    def get_observation(self) -> Observation:
        """Return the current observation as a plain dictionary."""

        return self.get_observation_model().to_dict()

    def get_observation_model(self) -> ObservationModel:
        """Return the current observation as a structured model."""

        state = self._require_state()
        if self.mode == "1d":
            assert isinstance(state, State1D)
            mass = state.mass if self.config.expose_mass_in_observation else None
            return Observation1D(
                position=state.position,
                velocity=state.velocity,
                acceleration=state.acceleration,
                previous_force=self._applied_force_1d(),
                step_count=self._step_count,
                mass=mass,
            )

        assert isinstance(state, State2D)
        mass = state.mass if self.config.expose_mass_in_observation else None
        return Observation2D(
            position=state.position,
            velocity=state.velocity,
            acceleration=state.acceleration,
            previous_force=self._applied_force_2d(),
            step_count=self._step_count,
            mass=mass,
        )

    def render(
        self,
        show_debug: bool | None = None,
        show_mass: bool | None = None,
        show_force_vector: bool | None = None,
    ) -> None:
        """Render the environment using lazily created Pygame UI objects."""

        if self._renderer is None or self._hud is None:
            from ui.hud import HudData, HudRenderer
            from ui.renderer import SandboxRenderer

            self._renderer = SandboxRenderer(self.config)
            self._hud = HudRenderer()

        debug_enabled = self.config.show_debug if show_debug is None else show_debug
        mass_visible = self.config.show_mass if show_mass is None else show_mass
        force_vector_enabled = (
            self.config.show_force_vector if show_force_vector is None else show_force_vector
        )

        self._renderer.begin_frame()
        self._renderer.render_scene(
            mode=self.mode,
            state=self.current_state,
            applied_action=self.current_applied_action,
            show_force_vector=force_vector_enabled,
        )
        hud_data = HudData(
            mode=self.mode,
            paused=self._done,
            done=self._done,
            frame_index=self._step_count,
            total_frames=self.config.episode_length_limit,
            show_debug=debug_enabled,
            show_mass=mass_visible,
            observation=self.get_observation_model(),
            applied_action=self.current_applied_action,
            true_mass=self.current_state.mass,
            status_message="Environment render view",
            episode_id="standalone-env",
            policy_name=None,
        )
        self._hud.draw(
            surface=self._renderer.surface,
            panel_rect=self._renderer.hud_rect,
            data=hud_data,
        )
        self._renderer.present()

    def close(self) -> None:
        """Release renderer resources if they were created."""

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._hud = None

    def _build_info(self, hit_boundary: bool | tuple[bool, bool]) -> dict[str, Any]:
        state = self._require_state()
        return {
            "mode": self.mode,
            "true_mass": state.mass,
            "step_count": self._step_count,
            "requested_force": self.current_requested_action.to_dict()["force"],
            "applied_force": self.current_applied_action.to_dict()["force"],
            "hit_boundary": hit_boundary,
            "done_reason": "episode_limit" if self._done else None,
        }

    def _coerce_action_1d(self, action: Any) -> Action1D:
        if isinstance(action, Action1D):
            return action
        if action is None:
            return Action1D(force=0.0)
        if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
            if len(action) != 1:
                raise ValueError(f"1D action expects one value, received {action!r}.")
            action = action[0]
        return Action1D(force=float(action))

    def _coerce_action_2d(self, action: Any) -> Action2D:
        if isinstance(action, Action2D):
            return action
        if action is None:
            return Action2D(force=(0.0, 0.0))
        if isinstance(action, Sequence) and not isinstance(action, (str, bytes)) and len(action) == 2:
            return Action2D(force=(float(action[0]), float(action[1])))
        raise ValueError(f"2D action expects a two-value sequence, received {action!r}.")

    def _zero_action(self, mode: Mode) -> ActionModel:
        if mode == "1d":
            return Action1D(force=0.0)
        return Action2D(force=(0.0, 0.0))

    def _applied_force_1d(self) -> float:
        applied = self._applied_action
        if isinstance(applied, Action1D):
            return applied.force
        return 0.0

    def _applied_force_2d(self) -> Vector2:
        applied = self._applied_action
        if isinstance(applied, Action2D):
            return applied.force
        return (0.0, 0.0)

    def _empty_boundary(self) -> bool | tuple[bool, bool]:
        return False if self.mode == "1d" else (False, False)

    def _require_state(self) -> StateModel:
        if self._state is None:
            raise RuntimeError("Environment state is unavailable. Did you reset the environment?")
        return self._state

    def _require_state_1d(self) -> State1D:
        state = self._require_state()
        if not isinstance(state, State1D):
            raise RuntimeError("Expected a 1D state but found a 2D state.")
        return state

    def _require_state_2d(self) -> State2D:
        state = self._require_state()
        if not isinstance(state, State2D):
            raise RuntimeError("Expected a 2D state but found a 1D state.")
        return state
