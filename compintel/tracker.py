"""Execution tracking helpers for CompIntel Research."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schemas import ExecutionCheckpoint, ExecutionTrackerSnapshot
from .audit_store import AuditStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ExecutionTracker:
    """Small audit trail for plan-driven implementation work."""

    objective: str
    current_phase: str = "week_1"
    status: str = "in_progress"
    checkpoints: list[ExecutionCheckpoint] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)
    audit_store: AuditStore | None = None

    def add_checkpoint(
        self,
        phase: str,
        status: str,
        owner: str | None = None,
        summary: str | None = None,
        evidence: list[str] | None = None,
    ) -> None:
        self.checkpoints.append(
            ExecutionCheckpoint(
                phase=phase,
                status=status,
                owner=owner,
                summary=summary,
                evidence=evidence or [],
            )
        )
        self.updated_at = _utc_now()
        if self.audit_store is not None:
            self.audit_store.append_checkpoint(
                {
                    "objective": self.objective,
                    "phase": phase,
                    "status": status,
                    "owner": owner,
                    "summary": summary,
                    "evidence": evidence or [],
                    "updated_at": self.updated_at,
                }
            )

    def record_decision(self, decision: str) -> None:
        self.decisions.append(decision)
        self.updated_at = _utc_now()

    def record_risk(self, risk: str) -> None:
        self.risks.append(risk)
        self.updated_at = _utc_now()

    def ask(self, question: str) -> None:
        self.pending_questions.append(question)
        self.updated_at = _utc_now()

    def snapshot(self) -> ExecutionTrackerSnapshot:
        snapshot = ExecutionTrackerSnapshot(
            objective=self.objective,
            current_phase=self.current_phase,
            status=self.status,
            checkpoints=list(self.checkpoints),
            pending_questions=list(self.pending_questions),
            decisions=list(self.decisions),
            risks=list(self.risks),
        )
        if self.audit_store is not None:
            self.audit_store.append_snapshot(snapshot.model_dump())
        return snapshot

    def as_dict(self) -> dict[str, Any]:
        return self.snapshot().model_dump()
