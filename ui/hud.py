"""Heads-up display renderer for episode replay."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from config import Mode
from models.action import Action1D, Action2D, ActionModel
from models.observation import Observation1D, Observation2D, ObservationModel

Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class HudData:
    """All information needed to draw the replay HUD."""

    mode: Mode
    paused: bool
    done: bool
    frame_index: int
    total_frames: int
    show_debug: bool
    show_mass: bool
    observation: ObservationModel
    applied_action: ActionModel
    true_mass: float
    status_message: str
    episode_id: str | None
    policy_name: str | None = None
    predicted_acceleration: float | None = None
    acceleration_error: float | None = None


class HudRenderer:
    """Draw a right-side panel with replay state and controls."""

    _PANEL_BG: Color = (250, 251, 253)
    _PANEL_BORDER: Color = (216, 222, 230)
    _TEXT: Color = (20, 28, 40)
    _SUBTLE: Color = (111, 123, 140)
    _ACCENT: Color = (44, 122, 199)
    _WARNING: Color = (185, 90, 36)

    def __init__(self) -> None:
        pygame.font.init()
        self.title_font = pygame.font.Font(None, 32)
        self.section_font = pygame.font.Font(None, 24)
        self.text_font = pygame.font.Font(None, 22)

    def draw(self, surface: pygame.Surface, panel_rect: pygame.Rect, data: HudData) -> None:
        """Render the entire HUD panel."""

        pygame.draw.rect(surface, self._PANEL_BG, panel_rect)
        pygame.draw.line(surface, self._PANEL_BORDER, panel_rect.topleft, panel_rect.bottomleft, 2)

        y = panel_rect.top + 20
        y = self._draw_text(surface, "Physics Replay", self.title_font, self._TEXT, panel_rect.left + 18, y)
        status_label = "Paused" if data.paused else "Playing"
        if data.done:
            status_label = "Replay complete"
        y = self._draw_text(
            surface,
            f"{data.mode.upper()} mode | {status_label}",
            self.text_font,
            self._ACCENT if not data.done else self._WARNING,
            panel_rect.left + 18,
            y + 4,
        )

        y = self._draw_section_header(surface, "Episode", panel_rect.left + 18, y + 18)
        y = self._draw_text(
            surface,
            f"ID: {data.episode_id or 'untracked'}",
            self.text_font,
            self._TEXT,
            panel_rect.left + 18,
            y,
        )
        if data.policy_name is not None:
            y = self._draw_text(
                surface,
                f"Policy: {data.policy_name}",
                self.text_font,
                self._TEXT,
                panel_rect.left + 18,
                y,
            )
        y = self._draw_text(
            surface,
            f"Frame: {data.frame_index} / {data.total_frames}",
            self.text_font,
            self._TEXT,
            panel_rect.left + 18,
            y,
        )

        if data.show_debug:
            y = self._draw_section_header(surface, "State", panel_rect.left + 18, y + 14)
            for line in self._state_lines(data.observation, data.applied_action, data.show_mass, data.true_mass):
                y = self._draw_text(surface, line, self.text_font, self._TEXT, panel_rect.left + 18, y)

            if data.predicted_acceleration is not None:
                y = self._draw_section_header(surface, "Model", panel_rect.left + 18, y + 14)
                y = self._draw_text(
                    surface,
                    f"Pred accel: {data.predicted_acceleration:8.3f}",
                    self.text_font,
                    self._TEXT,
                    panel_rect.left + 18,
                    y,
                )
                if data.acceleration_error is not None:
                    y = self._draw_text(
                        surface,
                        f"Accel error: {data.acceleration_error:8.3f}",
                        self.text_font,
                        self._TEXT,
                        panel_rect.left + 18,
                        y,
                    )

        y = self._draw_section_header(surface, "Controls", panel_rect.left + 18, y + 14)
        controls = [
            "Space play/pause",
            "Left / Right step back/forward",
            "R restart replay",
            "Tab debug | M mass",
            "Esc or Q quit",
        ]
        for line in controls:
            y = self._draw_text(surface, line, self.text_font, self._SUBTLE, panel_rect.left + 18, y)

        y = self._draw_section_header(surface, "Status", panel_rect.left + 18, y + 14)
        self._draw_wrapped_text(
            surface,
            data.status_message,
            self.text_font,
            self._TEXT,
            panel_rect.left + 18,
            y,
            panel_rect.width - 36,
        )

    def _state_lines(
        self,
        observation: ObservationModel,
        applied_action: ActionModel,
        show_mass: bool,
        true_mass: float,
    ) -> list[str]:
        if isinstance(observation, Observation1D):
            assert isinstance(applied_action, Action1D)
            lines = [
                f"Position: {observation.position:8.3f}",
                f"Velocity: {observation.velocity:8.3f}",
                f"Acceleration: {observation.acceleration:8.3f}",
                f"Applied force: {applied_action.force:8.3f}",
            ]
        else:
            assert isinstance(observation, Observation2D)
            assert isinstance(applied_action, Action2D)
            lines = [
                f"Position: ({observation.position[0]:7.3f}, {observation.position[1]:7.3f})",
                f"Velocity: ({observation.velocity[0]:7.3f}, {observation.velocity[1]:7.3f})",
                f"Acceleration: ({observation.acceleration[0]:7.3f}, {observation.acceleration[1]:7.3f})",
                f"Applied force: ({applied_action.force[0]:7.3f}, {applied_action.force[1]:7.3f})",
            ]
        lines.append(f"Mass: {true_mass:8.3f}" if show_mass else "Mass: hidden")
        return lines

    def _draw_section_header(
        self,
        surface: pygame.Surface,
        text: str,
        x: int,
        y: int,
    ) -> int:
        return self._draw_text(surface, text, self.section_font, self._ACCENT, x, y)

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        color: Color,
        x: int,
        y: int,
    ) -> int:
        rendered = font.render(text, True, color)
        surface.blit(rendered, (x, y))
        return y + rendered.get_height() + 4

    def _draw_wrapped_text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        color: Color,
        x: int,
        y: int,
        max_width: int,
    ) -> int:
        words = text.split()
        line = ""
        current_y = y
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if font.size(candidate)[0] <= max_width:
                line = candidate
                continue
            current_y = self._draw_text(surface, line, font, color, x, current_y)
            line = word
        if line:
            current_y = self._draw_text(surface, line, font, color, x, current_y)
        return current_y
