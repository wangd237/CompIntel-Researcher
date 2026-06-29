"""Runtime state contracts for CompIntel Research.

Every TypedDict in this module defines the data boundary between pipeline nodes.
When adding a new field, update both the TypedDict and the downstream nodes that
read it — LangGraph fan-out copies fields by name, not by type, so a missing
field means silent ``None`` at the next node.
"""

from __future__ import annotations

import operator
from typing import Any, TypedDict
from typing_extensions import NotRequired
from typing import Annotated


class ExecutionEvent(TypedDict, total=False):
    """A single event recorded by a pipeline node, appended to ``execution_log``."""
    node: str       # e.g. "intent_analyst", "competitor_profiler"
    event: str      # "started" | "completed" | "error"
    detail: str     # human-readable summary of what happened
    data: dict[str, Any]


class CompetitorProfile(TypedDict, total=False):
    """Per-competitor profile produced by CompetitorProfilerAgent."""
    name: str
    website: str
    summary: str
    pricing: list[dict[str, Any]]
    strengths: list[str]
    weaknesses: list[str]
    sources: list[dict[str, Any]]


class ReviewFeedback(TypedDict, total=False):
    """Quality gate output from ReviewerAgent.

    When ``approved`` is False the pipeline routes back to report_writer
    for revision (up to ``ReviewerAgent.MAX_RETRIES`` attempts).
    """
    score: int                          # 0-100 structural quality score
    approved: bool                      # True → route to rag_ingest
    retry_count: int                    # how many revision attempts so far
    issues: list[dict[str, Any]]        # specific issues found
    note: str                           # reviewer commentary


class CompIntelState(TypedDict, total=False):
    """Top-level pipeline state flowing through the 10-node LangGraph workflow.

    Fields marked ``Annotated[..., operator.add]`` are LangGraph reducers:
    each node's return value is appended, not replaced, so fan-out agents
    accumulate results safely.
    """

    # ── Input ────────────────────────────────────────────────────────────
    query: str

    # ── Intent Analyst output ────────────────────────────────────────────
    language: str
    intent: dict[str, Any]
    target: str
    market_segment: str
    competitors: list[dict[str, Any]]

    # ── Research Planner output ──────────────────────────────────────────
    research_questions: list[str]
    research_plan: dict[str, Any]

    # ── Competitor Profiler output (fan-out, Annotated=append) ───────────
    profiles: Annotated[list[dict[str, Any]], operator.add]

    # ── Curator output ───────────────────────────────────────────────────
    curated_profiles: list[dict[str, Any]]
    curator_evidence: dict[str, Any]

    # ── Analyst outputs ──────────────────────────────────────────────────
    market_analysis: dict[str, Any]
    swot_analysis: dict[str, Any]
    report: dict[str, Any]

    # ── Reviewer output ──────────────────────────────────────────────────
    review_feedback: ReviewFeedback

    # ── Bookkeeping ──────────────────────────────────────────────────────
    retry_count: int
    warnings: Annotated[list[str], operator.add]
    execution_log: Annotated[list[ExecutionEvent], operator.add]


class ExecutionTrackerState(TypedDict, total=False):
    """State snapshot used by ExecutionTracker for audit persistence."""
    objective: str
    current_phase: str
    status: str
    checkpoints: list[dict[str, Any]]
    pending_questions: list[str]
    decisions: list[dict[str, Any]]
    risks: list[dict[str, Any]]
    updated_at: str
