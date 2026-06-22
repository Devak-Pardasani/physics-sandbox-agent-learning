"""JSON export service for episode logs."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from models.transition import EpisodeRecord


class ExportService:
    """Persist episode logs to JSON files."""

    def __init__(self, export_dir: Path) -> None:
        self.export_dir = export_dir

    def export_episode(self, episode: EpisodeRecord) -> Path:
        """Write an episode log to the configured export directory."""

        self.export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"episode_{timestamp}_{episode.episode_id}_{episode.mode}.json"
        export_path = self.export_dir / filename
        with export_path.open("w", encoding="utf-8") as handle:
            json.dump(episode.to_dict(), handle, indent=2)
            handle.write("\n")
        return export_path
