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
    "intent_parsed",
    "profiling_start",
    "search_complete",
    "scrape_complete",
    "rag_complete",
    "profile_aggregated",
    "market_analysis_complete",
    "swot_complete",
    "report_ready",
    "review_passed",
    "error",
]


COMPINTEL_EVENT_TYPES: tuple[str, ...] = (
    "execution_started",
    "phase_started",
    "phase_completed",
    "analysis_ready",
    "checkpoint_written",
    "execution_completed",
    "execution_failed",
    "intent_parsed",
    "profiling_start",
    "search_complete",
    "scrape_complete",
    "rag_complete",
    "profile_aggregated",
    "market_analysis_complete",
    "swot_complete",
    "report_ready",
    "review_passed",
    "error",
)


COMPINTEL_PIPELINE_EVENT_TYPES: tuple[str, ...] = (
    "intent_parsed",
    "profiling_start",
    "search_complete",
    "scrape_complete",
    "rag_complete",
    "profile_aggregated",
    "market_analysis_complete",
    "swot_complete",
    "report_ready",
    "review_passed",
    "error",
)


COMPINTEL_EVENT_STAGE_LABELS: dict[str, str] = {
    "intent_parsed": "Intent analysis",
    "profiling_start": "Competitor profiling",
    "search_complete": "Search",
    "scrape_complete": "Scrape",
    "rag_complete": "RAG retrieval",
    "profile_aggregated": "Profile aggregation",
    "market_analysis_complete": "Market analysis",
    "swot_complete": "SWOT synthesis",
    "report_ready": "Report writing",
    "review_passed": "Review gate",
    "error": "Error",
}


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
