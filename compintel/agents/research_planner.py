"""Research planning for competitor profiling."""

from __future__ import annotations

from typing import Any

from .base import BaseCompIntelAgent


class ResearchPlannerAgent(BaseCompIntelAgent):
    """Create a structured plan for each competitor."""

    async def __call__(self, state: Any) -> dict[str, Any]:
        competitors = []
        if isinstance(state, dict):
            competitors = state.get("competitors") or []

        plan: dict[str, Any] = {}
        for competitor in competitors:
            name = competitor.get("name", "unknown")
            plan[name] = {
                "phases": [
                    {"phase": "company_overview", "queries": self._overview_queries(name)},
                    {"phase": "product_analysis", "queries": self._product_queries(name)},
                    {"phase": "pricing", "queries": self._pricing_queries(name)},
                    {"phase": "market_position", "queries": self._market_queries(name)},
                ],
                "search_strategy": {
                    "sources": ["official_website", "tech_media", "review_platforms", "financial_db"],
                    "max_results_per_phase": 5,
                },
            }

        return {
            "research_plan": plan,
            "execution_log": [
                {"node": "research_planner", "event": "completed", "detail": "competitor plan generated"}
            ],
        }

    def _overview_queries(self, name: str) -> list[str]:
        return [f"{name} company overview", f"{name} founding team", f"{name} mission"]

    def _product_queries(self, name: str) -> list[str]:
        return [f"{name} product features", f"{name} integrations", f"{name} use cases"]

    def _pricing_queries(self, name: str) -> list[str]:
        return [f"{name} pricing", f"{name} plans", f"{name} free tier"]

    def _market_queries(self, name: str) -> list[str]:
        return [f"{name} competitors", f"{name} market position", f"{name} reviews"]
