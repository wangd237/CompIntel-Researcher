"""Simple execution entrypoint for CompIntel Week 1."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from .audit_store import AuditStore
from .events import CompIntelEvent
from .graph import CompIntelGraph
from .settings import CompIntelSettings
from .tracker import ExecutionTracker


class CompIntelExecution:
    """Thin orchestration wrapper for the first runnable slice."""

    def __init__(self, model: str = "deepseek-chat") -> None:
        self.settings = CompIntelSettings.from_env()
        self.graph = CompIntelGraph(model=model)
        self.audit_store = AuditStore(Path(self.settings.report_store_path))
        self.tracker = ExecutionTracker(
            objective="CompIntel Research",
            audit_store=self.audit_store,
        )

    async def run_intent(self, query: str) -> dict[str, Any]:
        events: list[CompIntelEvent] = []
        events.append({
            "type": "execution_started",
            "message": "CompIntel intent run started",
            "phase": "intent_analyst",
            "data": {"query": query},
        })
        self.tracker.record_decision("Run Week 1 intent analysis")
        self.tracker.add_checkpoint(
            phase="intent_analyst",
            status="running",
            owner="agent/team",
            summary="Parsing query into target, competitors, and questions.",
            evidence=[query],
        )
        events.append({
            "type": "phase_started",
            "phase": "intent_analyst",
            "message": "Intent analysis started",
            "data": {"query": query},
        })
        response = await self.graph.run_competitor_pipeline(query)
        result = response.model_dump()
        events.append({
            "type": "analysis_ready",
            "phase": "intent_analyst",
            "message": "Structured intent captured",
            "data": {
                "target": result.get("intent", {}).get("target", ""),
                "market_segment": result.get("intent", {}).get("market_segment", ""),
                "competitors": len(result.get("competitors", [])),
            },
        })
        self.tracker.add_checkpoint(
            phase="intent_analyst",
            status="completed",
            owner="agent/team",
            summary="Intent analysis produced structured output.",
            evidence=[result.get("intent", {}).get("target", ""), result.get("intent", {}).get("market_segment", "")],
        )
        snapshot = self.tracker.as_dict()
        events.append({
            "type": "checkpoint_written",
            "phase": "intent_analyst",
            "message": "Snapshot written",
            "data": {
                "tracker": snapshot,
                "audit_path": str(self.audit_store.path),
            },
        })
        events.append({
            "type": "phase_completed",
            "phase": "intent_analyst",
            "message": "Intent analysis completed",
            "data": {
                "target": result.get("intent", {}).get("target", ""),
                "competitors": len(result.get("competitors", [])),
            },
        })
        events.append({
            "type": "execution_completed",
            "message": "CompIntel intent run completed",
            "phase": "intent_analyst",
            "data": {
                "tracker": snapshot,
                "audit_path": str(self.audit_store.path),
            },
        })
        return {
            "result": result,
            "tracker": snapshot,
            "audit_path": str(self.audit_store.path),
            "events": events,
        }
