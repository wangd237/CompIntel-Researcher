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

    When ``approved`` is False, the pipeline routes back to report_writer
    for revision (up to ``ReviewerAgent.MAX_RETRIES`` attempts).
    """

    score: int                          # 0-100 structural quality score
    approved: bool                      # True → route to rag_ingest
    retry_count: int                    # how many revision attempts so far
    issues: list[dict[str, Any]]        # specific issues found
    note: str                           # reviewer commentary


class CompIntelState(TypedDict, total=False):
    """Top-level pipeline state flowing through the 10-node LangGraph workflow.

    Fields marked ``Annotated[..., operator.add]`` are LangGraph reducers —
    each node's return value is appended, not replaced, so fan-out agents
    accumulate results safely.

    The pipeline order is:
        intent_analyst → research_planner → competitor_profiler (fan-out)
        → curator → market_analyst → swot_synthesizer → report_writer
        → editor → reviewer → [rag_ingest | report_writer (retry)]
    """

    # ── Input ────────────────────────────────────────────────────────────
    query: str                                              # user's original query

    # ── Intent Analyst output ────────────────────────────────────────────
    language: str                                           # "zh" | "en"
    intent: dict[str, Any]                                  # parsed target, segment, notes
    target: str                                             # the company / product being analyzed
    market_segment: str                                     # e.g. "SaaS", "automotive"
    competitors: list[dict[str, Any]]                      # [{name, website}, ...]

    # ── Research Planner output ──────────────────────────────────────────
    research_questions: list[str]                           # questions driving the analysis
    research_plan: dict[str, Any]                           # structured plan document

    # ── Competitor Profiler output (fan-out, Annotated=append) ───────────
    profiles: Annotated[list[dict[str, Any]], operator.add]  # raw profiles from fan-out

    # ── Curator output ───────────────────────────────────────────────────
    curated_profiles: list[dict[str, Any]]                  # cleaned + graded profiles
    curator_evidence: dict[str, Any]                        # evidence quality summary

    # ── Legacy / older path (Pydantic profile objects) ───────────────────
    competitor_profiles: Annotated[list[CompetitorProfile], operator.add]

    # ── Analyst outputs ──────────────────────────────────────────────────
    market_analysis: dict[str, Any]                         # market landscape
    swot_analysis: dict[str, Any]                           # per-competitor SWOT matrix
    report: dict[str, Any]                                  # final markdown report sections

    # ── Reviewer output ──────────────────────────────────────────────────
    review_feedback: ReviewFeedback                         # quality gate verdict

    # ── Bookkeeping ──────────────────────────────────────────────────────
    messages: list[dict[str, Any]]                          # internal message buffer
    retry_count: int                                        # reviewer revision counter
    warnings: list[str]                                     # non-fatal issues
    execution_log: Annotated[list[ExecutionEvent], operator.add]  # per-node event log
    errors: list[dict[str, Any]]                           # fatal errors
    metadata: dict[str, Any]                                # arbitrary metadata

    # ── Execution tracker fields ─────────────────────────────────────────
    phase: str                                              # current pipeline node name
    phase_status: str                                       # "running" | "completed" | "error"
    phase_owner: str                                        # "agent/team"
    phase_started_at: str                                   # ISO timestamp
    phase_updated_at: str                                   # ISO timestamp
    next_action: str                                        # next pipeline step
    blockers: list[str]                                     # what's blocking progress
    audit_notes: list[str]                                  # audit trail


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
