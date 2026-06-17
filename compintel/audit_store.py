"""Persistence helpers for CompIntel execution audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AuditStore:
    """Append-only JSONL store for execution snapshots."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            fh.write("\n")

    def append_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.append({"type": "snapshot", **snapshot})

    def append_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.append({"type": "checkpoint", **checkpoint})
