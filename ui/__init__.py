"""UI helpers for rendering and input."""

from __future__ import annotations

from typing import Any

__all__ = ["ReplayCommand", "HudData", "HudRenderer", "InputHandler", "SandboxRenderer"]


def __getattr__(name: str) -> Any:
    """Load Pygame-backed UI helpers only when they are requested."""

    if name in {"HudData", "HudRenderer"}:
        from .hud import HudData, HudRenderer

        return {"HudData": HudData, "HudRenderer": HudRenderer}[name]
    if name in {"InputHandler", "ReplayCommand"}:
        from .input_handler import InputHandler, ReplayCommand

        return {"InputHandler": InputHandler, "ReplayCommand": ReplayCommand}[name]
    if name == "SandboxRenderer":
        from .renderer import SandboxRenderer

        return SandboxRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
