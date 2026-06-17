"""Review gate for CompIntel Research."""

from __future__ import annotations

from typing import Any

from .base import BaseCompIntelAgent


class ReviewerAgent(BaseCompIntelAgent):
    MAX_RETRIES = 3

    async def __call__(self, state: Any) -> dict[str, Any]:
        report = {}
        retry_count = 0
        if isinstance(state, dict):
            report = state.get("report") or {}
            retry_count = int((state.get("review_feedback") or {}).get("retry_count", 0))

        score = 7 if report.get("sections") else 4
        approved = score >= 7 or retry_count >= self.MAX_RETRIES
        feedback = {
            "score": score,
            "approved": approved,
            "retry_count": retry_count,
            "issues": [] if approved else [{"type": "missing_sections", "detail": "report is incomplete"}],
            "note": "Approved" if approved else "Needs revision",
        }
        return {
            "review_feedback": feedback,
            "execution_log": [
                {"node": "reviewer", "event": "completed", "detail": "approved" if approved else "revision requested"}
            ],
        }
