"""Pydantic contracts for CompIntel Research inputs and outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompetitorCandidate(BaseModel):
    name: str = Field(..., description="Competitor name")
    website: str | None = Field(default=None, description="Official website URL")
    rationale: str | None = Field(default=None, description="Why this competitor matters")


class CompetitorProfileSchema(BaseModel):
    name: str
    website: str | None = None
    summary: str | None = None
    search_results: list[dict[str, Any]] = Field(default_factory=list)
    scraped_content: list[dict[str, Any]] = Field(default_factory=list)
    rag_context: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class IntentAnalysisResponse(BaseModel):
    target: str = Field(..., description="Primary company or product under analysis")
    market_segment: str = Field(..., description="Market or category being analyzed")
    competitors: list[CompetitorCandidate] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CompIntelAnalyzeRequest(BaseModel):
    query: str = Field(..., description="Natural-language research request")
    max_competitors: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="zh-CN")


class CompIntelAnalyzeResponse(BaseModel):
    query: str
    intent: IntentAnalysisResponse | None = None
    competitors: list[CompetitorCandidate] = Field(default_factory=list)
    profiles: list[CompetitorProfileSchema] = Field(default_factory=list)
    report: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class ExecutionCheckpoint(BaseModel):
    phase: str
    status: str
    owner: str | None = None
    summary: str | None = None
    evidence: list[str] = Field(default_factory=list)


class ExecutionTrackerSnapshot(BaseModel):
    objective: str
    current_phase: str
    status: str
    checkpoints: list[ExecutionCheckpoint] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
