"""Search worker for competitor profiling."""

from __future__ import annotations

from typing import Any

from .base import BaseCompIntelAgent


class SearchWorker(BaseCompIntelAgent):
    async def __call__(self, state: Any) -> dict[str, Any]:
        competitor = {}
        questions: list[str] = []
        if isinstance(state, dict):
            competitor = state.get("competitor") or {}
            questions = state.get("research_questions") or []

        name = competitor.get("name", "unknown")
        results = [
            {
                "title": f"{name} search result {idx + 1}",
                "url": f"https://example.com/{name.lower().replace(' ', '-')}/{idx + 1}",
                "snippet": question,
                "source": "placeholder",
            }
            for idx, question in enumerate(questions[:5])
        ]
        return {
            "search_results": results,
            "execution_log": [
                {"node": "search_worker", "event": "completed", "detail": f"{len(results)} results"}
            ],
        }
