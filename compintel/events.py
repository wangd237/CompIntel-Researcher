"""Event types for CompIntel Research streaming and audit."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


EventType = Literal[
    "execution_started",
    "phase_started",
    "phase_completed",
    "analysis_ready",
    "checkpoint_written",
    "execution_completed",
    "execution_failed",
]


class CompIntelEventData(TypedDict, total=False):
    query: str
    target: str
    market_segment: str
    competitors: int
    tracker: dict[str, Any]
    audit_path: str


class CompIntelEvent(TypedDict, total=False):
    type: EventType
    phase: str
    message: str
    data: CompIntelEventData
