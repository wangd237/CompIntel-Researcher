"""Simple execution entrypoint for CompIntel Week 1."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from pathlib import Path

from .audit_store import AuditStore
from .events import CompIntelEvent
from .graph import CompIntelGraph
from .settings import CompIntelSettings
from .tracker import ExecutionTracker

logger = logging.getLogger(__name__)


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

    async def run_intent_streaming(self, query: str):
        """Streaming variant: yields progress events as each pipeline stage completes.

        Uses LangGraph's ``astream(stream_mode="values")`` to get the full state
        after each superstep.  Extracts the current phase from ``execution_log``
        and yields a progress event.  The final event is ``analysis_ready`` with
        the full structured result.
        """
        import time

        from .bundle import generate_delivery_bundle
        from .schemas import CompIntelAnalyzeResponse, CompetitorProfileSchema

        PHASE_LABELS: dict[str, str] = {
            "intent_analyst": "Analyzing query and identifying competitors",
            "research_planner": "Planning research strategy",
            "competitor_profiler": "Profiling competitors (search + scrape + RAG)",
            "curator": "Cleaning profiles and grading evidence quality",
            "market_analyst": "Analyzing market landscape",
            "swot_synthesizer": "Building SWOT analysis",
            "report_writer": "Writing competitive intelligence report",
            "editor": "Editorial review",
            "reviewer": "Quality gate review",
            "rag_ingest": "Saving analysis to memory",
        }

        yield {
            "type": "execution_started",
            "phase": "startup",
            "message": "CompIntel analysis started",
            "data": {"query": query[:200]},
        }

        t0 = time.monotonic()
        last_phase: str | None = None
        final_state: dict[str, Any] = {}

        config = {
            "configurable": {
                "thread_id": f"compintel:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
            }
        }

        try:
            async for state in self.graph.app.astream(
                {
                    "query": query,
                    "profiles": [],
                    "execution_log": [],
                    "retry_count": 0,
                },
                config,
                stream_mode="values",
            ):
                final_state = state
                logs = state.get("execution_log", [])
                if not logs:
                    continue
                latest = logs[-1] if isinstance(logs, list) else None
                if not isinstance(latest, dict):
                    continue
                node = latest.get("node", "")
                if node and node != last_phase and node in PHASE_LABELS:
                    last_phase = node
                    elapsed = time.monotonic() - t0
                    yield {
                        "type": "phase_completed",
                        "phase": node,
                        "message": f"{PHASE_LABELS[node]} ({elapsed:.0f}s)",
                        "data": {"elapsed_s": round(elapsed, 1)},
                    }
        except Exception as exc:
            logger.exception("Streaming pipeline failed")
            yield {
                "type": "execution_failed",
                "phase": last_phase or "unknown",
                "message": f"Analysis failed: {str(exc)[:200]}",
                "data": {"error": str(exc)},
            }
            return

        elapsed = time.monotonic() - t0

        try:
            self.tracker.add_checkpoint(
                phase="completed",
                status="completed",
                owner="agent/team",
                summary=f"Streaming analysis completed in {elapsed:.0f}s",
                evidence=[query],
            )
        except Exception:
            pass

        result = CompIntelAnalyzeResponse(
            query=query,
            intent=final_state.get("intent"),
            competitors=final_state.get("competitors", []),
            profiles=[
                CompetitorProfileSchema(**p)
                for p in (final_state.get("profiles") or [])
            ],
            report={
                "research_plan": final_state.get("research_plan", {}),
                "market_analysis": final_state.get("market_analysis", {}),
                "swot_analysis": final_state.get("swot_analysis", {}),
                "report": final_state.get("report", {}),
                "review_feedback": final_state.get("review_feedback", {}),
                "execution_log": final_state.get("execution_log", []),
                "curator_evidence": final_state.get("curator_evidence", {}),
            },
            warnings=final_state.get("warnings", []),
        ).model_dump()

        tracker_dict = self.tracker.as_dict()
        bundle = generate_delivery_bundle({
            "result": result,
            "tracker": tracker_dict,
            "audit_path": str(self.audit_store.path) if self.audit_store else "",
        })

        yield {
            "type": "analysis_ready",
            "message": "Analysis complete",
            "data": {
                "mode": "streaming",
                "event_count": "streamed",
                "result": result,
                "tracker": tracker_dict,
                "audit_path": str(self.audit_store.path) if self.audit_store else "",
                **bundle,
            },
        }
