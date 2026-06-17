"""Scrape worker for competitor profiling."""

from __future__ import annotations

from typing import Any

from .base import BaseCompIntelAgent


class ScrapeWorker(BaseCompIntelAgent):
    async def __call__(self, state: Any) -> dict[str, Any]:
        competitor = {}
        if isinstance(state, dict):
            competitor = state.get("competitor") or {}

        name = competitor.get("name", "unknown")
        scraped = [
            {
                "url": competitor.get("website") or f"https://example.com/{name.lower().replace(' ', '-')}",
                "title": f"{name} overview",
                "content": f"Placeholder scraped content for {name}.",
            }
        ]
        return {
            "scraped_content": scraped,
            "execution_log": [
                {"node": "scrape_worker", "event": "completed", "detail": "placeholder scrape completed"}
            ],
        }
