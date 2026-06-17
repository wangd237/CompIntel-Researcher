"""Progress summary rendering for CompIntel Research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProgressSummaryFormatter:
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, tracker: dict[str, Any], events: list[dict[str, Any]] | None = None) -> str:
        checkpoints = tracker.get("checkpoints", [])
        last_checkpoint = checkpoints[-1] if checkpoints else {}
        lines = [
            "# CompIntel Progress Summary",
            "",
            f"- Objective: {tracker.get('objective', 'unknown')}",
            f"- Phase: {tracker.get('current_phase', 'unknown')}",
            f"- Status: {tracker.get('status', 'unknown')}",
            f"- Decisions: {len(tracker.get('decisions', []))}",
            f"- Risks: {len(tracker.get('risks', []))}",
            f"- Pending Questions: {len(tracker.get('pending_questions', []))}",
            "",
            "## Latest Checkpoint",
            self._format_checkpoint(last_checkpoint),
            "",
            "## Event Trail",
        ]
        for event in events or []:
            lines.append(f"- {event.get('type', 'event')}: {event.get('message', '')}")
        return "\n".join(lines)

    def save(self, tracker: dict[str, Any], events: list[dict[str, Any]] | None = None, filename: str = "compintel_progress.md") -> Path:
        path = self.output_dir / filename
        path.write_text(self.render(tracker, events), encoding="utf-8")
        return path

    def _format_checkpoint(self, checkpoint: Any) -> str:
        if not isinstance(checkpoint, dict) or not checkpoint:
            return "- empty"
        return "\n".join(
            [
                f"- phase: {checkpoint.get('phase', 'unknown')}",
                f"- status: {checkpoint.get('status', 'unknown')}",
                f"- owner: {checkpoint.get('owner', 'unknown')}",
                f"- summary: {checkpoint.get('summary', '')}",
            ]
        )
