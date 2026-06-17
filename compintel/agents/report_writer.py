"""Report writing for CompIntel Research."""

from __future__ import annotations

from typing import Any

from .base import BaseCompIntelAgent


class ReportWriterAgent(BaseCompIntelAgent):
    async def __call__(self, state: Any) -> dict[str, Any]:
        query = "unknown"
        intent = {}
        profiles = []
        market_analysis = {}
        swot_analysis = {}
        if isinstance(state, dict):
            query = state.get("query") or query
            intent = state.get("intent") or {}
            profiles = state.get("profiles") or []
            market_analysis = state.get("market_analysis") or {}
            swot_analysis = state.get("swot_analysis") or {}

        report = {
            "title": f"CompIntel Report: {query}",
            "executive_summary": f"Analysis for {intent.get('target', 'unknown')}",
            "profiles": profiles,
            "market_analysis": market_analysis,
            "swot_analysis": swot_analysis,
            "sections": [
                "Executive Summary",
                "Competitor Profiles",
                "Market Analysis",
                "SWOT",
            ],
        }
        return {
            "report": report,
            "execution_log": [
                {"node": "report_writer", "event": "completed", "detail": "report drafted"}
            ],
        }
