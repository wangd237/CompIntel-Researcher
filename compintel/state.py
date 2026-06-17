"""Runtime state contracts for CompIntel Research."""

from __future__ import annotations

from typing import Any, TypedDict
from typing_extensions import NotRequired


class ExecutionEvent(TypedDict, total=False):
    node: str
    event: str
    detail: str
    data: dict[str, Any]


class CompetitorProfile(TypedDict, total=False):
    name: str
    website: str
    summary: str
    pricing: list[dict[str, Any]]
    strengths: list[str]
    weaknesses: list[str]
    sources: list[dict[str, Any]]


class ReviewFeedback(TypedDict, total=False):
    score: int
    approved: bool
    retry_count: int
    issues: list[dict[str, Any]]
    note: str


class CompIntelState(TypedDict, total=False):
    query: str
    market_segment: str
    status: str
    competitors: list[dict[str, Any]]
    research_questions: list[str]
    research_plan: dict[str, Any]
    competitor_profiles: list[CompetitorProfile]
    market_analysis: dict[str, Any]
    swot_analysis: dict[str, Any]
    report: dict[str, Any]
    review_feedback: ReviewFeedback
    messages: list[dict[str, Any]]
    execution_log: list[ExecutionEvent]
    errors: list[dict[str, Any]]
    metadata: dict[str, Any]
    checkpoint_id: NotRequired[str]
    phase: str
    phase_status: str
    phase_owner: str
    phase_started_at: str
    phase_updated_at: str
    next_action: str
    blockers: list[str]
    audit_notes: list[str]


class CompetitorProfilerState(TypedDict, total=False):
    competitor: dict[str, Any]
    research_questions: list[str]
    search_results: list[dict[str, Any]]
    scraped_content: list[dict[str, Any]]
    rag_context: list[dict[str, Any]]
    profile: CompetitorProfile
    errors: list[dict[str, Any]]
    execution_log: list[ExecutionEvent]


class ExecutionTrackerState(TypedDict, total=False):
    objective: str
    current_phase: str
    status: str
    checkpoints: list[dict[str, Any]]
    pending_questions: list[str]
    decisions: list[dict[str, Any]]
    risks: list[dict[str, Any]]
    updated_at: str
