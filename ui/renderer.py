"""Minimal canvas renderer for the sandbox desktop app."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from config import SandboxConfig
from models.action import Action1D, Action2D, ActionModel
from models.state import State1D, State2D, StateModel
from utils.coordinate_mapper import CoordinateMapper
from utils.vectors import vector_is_near_zero

Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Layout:
    """Screen layout for the simulation canvas and HUD."""

    scene_rect: pygame.Rect
    hud_rect: pygame.Rect


class SandboxRenderer:
    """Render the simulation scene while leaving HUD drawing to a separate module."""

    _BACKGROUND: Color = (241, 244, 248)
    _SCENE_BG: Color = (255, 255, 255)
    _SCENE_BORDER: Color = (214, 221, 230)
    _TRACK: Color = (64, 78, 96)
    _PARTICLE: Color = (47, 124, 200)
    _FORCE: Color = (222, 94, 52)
    _CENTER_MARK: Color = (145, 157, 173)
    _DONE_BG: Color = (32, 42, 58)
    _DONE_TEXT: Color = (250, 252, 255)

    def __init__(self, config: SandboxConfig) -> None:
        pygame.init()
        pygame.font.init()
        self.config = config
        self._status_font = pygame.font.Font(None, 26)
        self._layout = self._build_layout(config)
        pygame.display.set_caption(f"{config.window_title} [{config.mode.upper()}]")
        self._surface = pygame.display.set_mode((config.screen_width, config.screen_height))

    @property
    def surface(self) -> pygame.Surface:
        """Return the current display surface."""

        return self._surface

    @property
    def scene_rect(self) -> pygame.Rect:
        """Return the scene drawing region."""

        return self._layout.scene_rect

    @property
    def hud_rect(self) -> pygame.Rect:
        """Return the HUD drawing region."""

        return self._layout.hud_rect

    def apply_config(self, config: SandboxConfig) -> None:
        """Update renderer configuration after a mode or window change."""

        self.config = config
        self._layout = self._build_layout(config)
        pygame.display.set_caption(f"{config.window_title} [{config.mode.upper()}]")
        if self._surface.get_size() != (config.screen_width, config.screen_height):
            self._surface = pygame.display.set_mode((config.screen_width, config.screen_height))

    def begin_frame(self) -> None:
        """Clear the window before drawing the next frame."""

        self._surface.fill(self._BACKGROUND)
        pygame.draw.rect(self._surface, self._SCENE_BG, self.scene_rect, border_radius=18)
        pygame.draw.rect(self._surface, self._SCENE_BORDER, self.scene_rect, width=2, border_radius=18)

    def render_scene(
        self,
        mode: str,
        state: StateModel,
        applied_action: ActionModel,
        show_force_vector: bool,
        done: bool = False,
    ) -> None:
        """Draw the current simulation scene."""

        if mode == "1d":
            assert isinstance(state, State1D)
            assert isinstance(applied_action, Action1D)
            self._render_1d(state, applied_action, show_force_vector)
        else:
            assert isinstance(state, State2D)
            assert isinstance(applied_action, Action2D)
            self._render_2d(state, applied_action, show_force_vector)

        if done:
            self._draw_done_badge()

    def present(self) -> None:
        """Flip the display buffer."""

        pygame.display.flip()

    def close(self) -> None:
        """Close the Pygame display."""

        pygame.quit()

    def _render_1d(
        self,
        state: State1D,
        applied_action: Action1D,
        show_force_vector: bool,
    ) -> None:
        track_y = self.scene_rect.centery
        left = self.scene_rect.left + 36
        right = self.scene_rect.right - 36

        pygame.draw.line(self._surface, self._TRACK, (left, track_y), (right, track_y), 4)
        pygame.draw.line(
            self._surface,
            self._CENTER_MARK,
            (self.scene_rect.centerx, track_y - 24),
            (self.scene_rect.centerx, track_y + 24),
            2,
        )

        mapper = CoordinateMapper(
            world_x=(self.config.world_bounds_x.minimum, self.config.world_bounds_x.maximum),
            world_y=(0.0, 1.0),
            screen_left=left,
            screen_top=track_y,
            screen_width=right - left,
            screen_height=1,
        )
        particle_x = mapper.map_x(state.position)
        particle_y = track_y
        pygame.draw.circle(
            self._surface,
            self._PARTICLE,
            (particle_x, particle_y),
            self.config.object_radius,
        )

        if show_force_vector and abs(applied_action.force) > 1e-6:
            arrow_length = 130.0 * (applied_action.force / max(self.config.max_force, 1e-6))
            start = (particle_x, particle_y - 46)
            end = (int(round(particle_x + arrow_length)), particle_y - 46)
            self._draw_arrow(start, end, self._FORCE)

    def _render_2d(
        self,
        state: State2D,
        applied_action: Action2D,
        show_force_vector: bool,
    ) -> None:
        bounds = self.scene_rect.inflate(-70, -70)
        pygame.draw.rect(self._surface, self._TRACK, bounds, width=3, border_radius=14)

        mapper = CoordinateMapper(
            world_x=(self.config.world_bounds_x.minimum, self.config.world_bounds_x.maximum),
            world_y=(self.config.world_bounds_y.minimum, self.config.world_bounds_y.maximum),
            screen_left=bounds.left,
            screen_top=bounds.top,
            screen_width=bounds.width,
            screen_height=bounds.height,
        )
        particle_x, particle_y = mapper.map_point(state.position)
        pygame.draw.circle(
            self._surface,
            self._PARTICLE,
            (particle_x, particle_y),
            self.config.object_radius,
        )

        if show_force_vector and not vector_is_near_zero(applied_action.force):
            scale = 100.0 / max(self.config.max_force, 1e-6)
            end = (
                int(round(particle_x + applied_action.force[0] * scale)),
                int(round(particle_y - applied_action.force[1] * scale)),
            )
            self._draw_arrow((particle_x, particle_y), end, self._FORCE)

    def _draw_done_badge(self) -> None:
        badge = pygame.Rect(self.scene_rect.left + 20, self.scene_rect.top + 20, 270, 40)
        pygame.draw.rect(self._surface, self._DONE_BG, badge, border_radius=10)
        label = self._status_font.render("Episode finished. Reset or export.", True, self._DONE_TEXT)
        self._surface.blit(label, (badge.left + 12, badge.top + 10))

    def _draw_arrow(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        color: Color,
    ) -> None:
        pygame.draw.line(self._surface, color, start, end, width=4)
        direction = pygame.math.Vector2(end[0] - start[0], end[1] - start[1])
        if direction.length_squared() == 0.0:
            return
        tip = pygame.math.Vector2(end)
        unit = direction.normalize()
        base = tip - unit * 14.0
        perpendicular = pygame.math.Vector2(-unit.y, unit.x) * 6.0
        points = [
            (tip.x, tip.y),
            (base.x + perpendicular.x, base.y + perpendicular.y),
            (base.x - perpendicular.x, base.y - perpendicular.y),
        ]
        pygame.draw.polygon(self._surface, color, points)

    def _build_layout(self, config: SandboxConfig) -> Layout:
        hud_width = max(300, min(360, config.screen_width // 3))
        scene_rect = pygame.Rect(16, 16, config.screen_width - hud_width - 32, config.screen_height - 32)
        hud_rect = pygame.Rect(scene_rect.right + 16, 0, hud_width, config.screen_height)
        return Layout(scene_rect=scene_rect, hud_rect=hud_rect)
