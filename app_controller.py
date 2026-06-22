"""Replay-only desktop controller for inspecting logged episodes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from config import SandboxConfig
from models.action import Action1D, Action2D, ActionModel
from models.observation import ObservationModel
from models.state import StateModel
from models.transition import EpisodeRecord
from training.dataset import build_history_window_1d
from ui.hud import HudData, HudRenderer
from ui.input_handler import InputHandler
from ui.renderer import SandboxRenderer

if TYPE_CHECKING:
    from training.checkpoint import InferenceBundle


class ReplayController:
    """Run a Pygame replay UI for previously collected episodes."""

    def __init__(
        self,
        episode: EpisodeRecord,
        config: SandboxConfig,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.episode = episode
        self.config = config
        self.renderer = SandboxRenderer(config)
        self.hud = HudRenderer()
        self.input_handler = InputHandler()
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.show_debug = config.show_debug
        self.show_mass = config.show_mass
        self.frame_index = 0
        self.bundle: InferenceBundle | None = None
        self.status_message = "Replay loaded."

        if checkpoint_path is not None and episode.mode == "1d":
            from training.checkpoint import load_checkpoint

            self.bundle = load_checkpoint(checkpoint_path, device="cpu")
            self.status_message = f"Replay loaded with model overlay from {checkpoint_path.name}."
        elif checkpoint_path is not None:
            self.status_message = "Model overlay is only available for 1D episodes."

    def run(self) -> None:
        """Run the replay loop."""

        try:
            while self.running:
                self.tick()
                self.clock.tick(self.config.target_fps)
        finally:
            self.close()

    def tick(self, events: list[pygame.event.Event] | None = None) -> None:
        """Process one replay frame."""

        if events is None:
            events = pygame.event.get()
        command = self.input_handler.process(events)

        if command.quit_requested:
            self.running = False
            return
        if command.toggle_pause_requested:
            self.paused = not self.paused
        if command.toggle_debug_requested:
            self.show_debug = not self.show_debug
        if command.toggle_mass_requested:
            self.show_mass = not self.show_mass
        if command.restart_requested:
            self.frame_index = 0
            self.paused = False
            self.status_message = "Replay restarted."
        if command.step_backward_requested:
            self.paused = True
            self.frame_index = max(0, self.frame_index - 1)
        if command.step_forward_requested:
            self.paused = True
            self.frame_index = min(len(self.episode.transitions), self.frame_index + 1)
        elif not self.paused and self.frame_index < len(self.episode.transitions):
            self.frame_index += 1

        if self.frame_index >= len(self.episode.transitions):
            self.paused = True
            self.status_message = "Replay complete. Press R to restart."

        self.render_frame()

    def render_frame(self) -> None:
        """Render the current replay frame."""

        self.renderer.begin_frame()
        self.renderer.render_scene(
            mode=self.episode.mode,
            state=self.current_state,
            applied_action=self.current_action,
            show_force_vector=self.config.show_force_vector,
            done=self.frame_index >= len(self.episode.transitions),
        )
        predicted_acceleration, acceleration_error = self._prediction_overlay()
        hud_data = HudData(
            mode=self.episode.mode,
            paused=self.paused,
            done=self.frame_index >= len(self.episode.transitions),
            frame_index=self.frame_index,
            total_frames=len(self.episode.transitions),
            show_debug=self.show_debug,
            show_mass=self.show_mass,
            observation=self.current_observation,
            applied_action=self.current_action,
            true_mass=self.episode.initial_state.mass,
            status_message=self.status_message,
            episode_id=self.episode.episode_id,
            policy_name=self.episode.policy_name,
            predicted_acceleration=predicted_acceleration,
            acceleration_error=acceleration_error,
        )
        self.hud.draw(self.renderer.surface, self.renderer.hud_rect, hud_data)
        self.renderer.present()

    @property
    def current_state(self) -> StateModel:
        if self.frame_index == 0:
            return self.episode.initial_state
        return self.episode.transitions[self.frame_index - 1].next_state

    @property
    def current_observation(self) -> ObservationModel:
        if self.frame_index == 0:
            return self.episode.initial_observation
        return self.episode.transitions[self.frame_index - 1].observation_after

    @property
    def current_action(self) -> ActionModel:
        if self.frame_index == 0:
            if self.episode.mode == "1d":
                return Action1D(force=0.0)
            return Action2D(force=(0.0, 0.0))
        return self.episode.transitions[self.frame_index - 1].applied_action

    def close(self) -> None:
        """Release renderer resources."""

        self.renderer.close()

    def _prediction_overlay(self) -> tuple[float | None, float | None]:
        if self.bundle is None or self.episode.mode != "1d" or self.frame_index == 0:
            return None, None

        transition_index = self.frame_index - 1
        if transition_index < self.bundle.history_length - 1:
            return None, None

        features = build_history_window_1d(
            self.episode.transitions,
            end_index=transition_index,
            history_length=self.bundle.history_length,
        )
        prediction = self.bundle.predict_acceleration(features)
        true_acceleration = self.episode.transitions[transition_index].next_state.acceleration
        return prediction, abs(prediction - true_acceleration)
