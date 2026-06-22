"""Keyboard input mapping for episode replay."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(slots=True)
class ReplayCommand:
    """Replay controls requested during one frame."""

    quit_requested: bool = False
    toggle_pause_requested: bool = False
    step_forward_requested: bool = False
    step_backward_requested: bool = False
    restart_requested: bool = False
    toggle_debug_requested: bool = False
    toggle_mass_requested: bool = False


class InputHandler:
    """Translate keyboard input into replay commands."""

    def process(self, events: list[pygame.event.Event]) -> ReplayCommand:
        """Convert replay keyboard input into a single-frame command."""

        command = ReplayCommand()
        for event in events:
            if event.type == pygame.QUIT:
                command.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    command.quit_requested = True
                elif event.key == pygame.K_SPACE:
                    command.toggle_pause_requested = True
                elif event.key == pygame.K_RIGHT:
                    command.step_forward_requested = True
                elif event.key == pygame.K_LEFT:
                    command.step_backward_requested = True
                elif event.key == pygame.K_r:
                    command.restart_requested = True
                elif event.key == pygame.K_TAB:
                    command.toggle_debug_requested = True
                elif event.key == pygame.K_m:
                    command.toggle_mass_requested = True
        return command
