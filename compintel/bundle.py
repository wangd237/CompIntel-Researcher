"""Artifact bundle writer for CompIntel Research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any

from .export import MarkdownFormatter
from .progress import ProgressSummaryFormatter


@dataclass(slots=True)
class BundleWriter:
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any], bundle_name: str | None = None) -> Path:
        bundle_name = bundle_name or self._default_bundle_name()
        bundle_dir = self.output_dir / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        report_path = MarkdownFormatter(output_dir=bundle_dir).save(payload, "report.md")
        progress_path = ProgressSummaryFormatter(output_dir=bundle_dir).save(
            payload.get("tracker", {}),
            payload.get("events", []),
            "progress.md",
        )
        snapshot_path = bundle_dir / "snapshot.json"
        snapshot_path.write_text(self._render_snapshot(payload), encoding="utf-8")

        manifest_path = bundle_dir / "manifest.txt"
        manifest_path.write_text(
            "\n".join(
                [
                    f"generated_at={datetime.now(timezone.utc).isoformat()}",
                    f"report={report_path.name}",
                    f"progress={progress_path.name}",
                    f"snapshot={snapshot_path.name}",
                    f"bundle={bundle_dir.name}",
                ]
            ),
            encoding="utf-8",
        )
        return bundle_dir

    def _default_bundle_name(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        return f"compintel_bundle_{timestamp}_{uuid4().hex[:6]}"

    def _render_snapshot(self, payload: dict[str, Any]) -> str:
        from json import dumps

        return dumps(
            {
                "result": payload.get("result", {}),
                "tracker": payload.get("tracker", {}),
                "events": payload.get("events", []),
                "audit_path": payload.get("audit_path"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def generate_delivery_bundle(payload: dict[str, Any], output_dir: Path | str = "outputs", bundle_name: str | None = None) -> dict[str, str]:
    writer = BundleWriter(output_dir=Path(output_dir))
    bundle_dir = writer.write(payload, bundle_name=bundle_name)
    return {
        "bundle_path": str(bundle_dir),
        "report_path": str(bundle_dir / "report.md"),
        "progress_path": str(bundle_dir / "progress.md"),
        "snapshot_path": str(bundle_dir / "snapshot.json"),
        "manifest_path": str(bundle_dir / "manifest.txt"),
    }
