"""Markdown report formatting for CompIntel Research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MarkdownFormatter:
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, payload: dict[str, Any]) -> str:
        result = payload.get("result") or {}
        intent = result.get("intent") or {}
        profiles = result.get("profiles") or []
        report = result.get("report") or {}

        lines = [
            f"# CompIntel Research Report: {intent.get('target', 'unknown')}",
            "",
            f"## Market Segment",
            str(intent.get("market_segment", "unknown")),
            "",
            "## Competitors",
        ]
        for competitor in result.get("competitors", []):
            lines.append(f"- {competitor.get('name', 'unknown')}")
        lines.extend(
            [
                "",
                "## Profiles",
            ]
        )
        for profile in profiles:
            lines.append(f"- {profile.get('name', 'unknown')}: {profile.get('summary', '')}")
        lines.extend(
            [
                "",
                "## Market Analysis",
                self._render_block(report.get("market_analysis", {})),
                "",
                "## SWOT",
                self._render_block(report.get("swot_analysis", {})),
                "",
                "## Review",
                self._render_block(report.get("review_feedback", {})),
                "",
            ]
        )
        return "\n".join(lines)

    def save(self, payload: dict[str, Any], filename: str = "compintel_report.md") -> Path:
        path = self.output_dir / filename
        path.write_text(self.render(payload), encoding="utf-8")
        return path

    def _render_block(self, value: Any) -> str:
        if isinstance(value, dict):
            return "\n".join(f"- {key}: {item}" for key, item in value.items()) or "- empty"
        return str(value)
