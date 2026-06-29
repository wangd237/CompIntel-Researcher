"""Competitor profiling orchestration — direct asyncio.gather, no subgraph."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import BaseCompIntelAgent
from .rag_retriever import RAGRetriever
from .scrape_worker import ScrapeWorker
from .search_worker import SearchWorker

logger = logging.getLogger(__name__)


class CompetitorProfilerAgent(BaseCompIntelAgent):
    """Profile a single competitor via parallel Search + Scrape + RAG.

    Workers run concurrently via ``asyncio.gather`` with per-worker
    exception isolation — one worker failing does not lose data from
    the other two.
    """

    def __init__(self, model: str = "deepseek-chat") -> None:
        super().__init__(model=model)
        self.search_worker = SearchWorker(model=model)
        self.scrape_worker = ScrapeWorker(model=model)
        self.rag_retriever = RAGRetriever(model=model)

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        competitor = s.competitor
        market_segment = s.market_segment
        research_questions = s.research_questions

        search, scrape, rag = await asyncio.gather(
            self._safe_search(competitor, research_questions),
            self._safe_scrape(competitor, market_segment),
            self._safe_rag(competitor, market_segment),
        )

        name = competitor.get("name", "unknown") if isinstance(competitor, dict) else "unknown"
        website = competitor.get("website") if isinstance(competitor, dict) else None

        profile = {
            "name": name,
            "website": website,
            "summary": self._build_profile_summary(name, search, scrape, rag),
            "search_results": search.get("search_results", []),
            "scraped_content": scrape.get("scraped_content", []),
            "rag_context": rag.get("rag_context", []),
            "sources": self._extract_sources(search, scrape, rag),
        }

        return {
            "profiles": [profile],
            "execution_log": [
                {"node": "competitor_profiler", "event": "completed", "detail": name}
            ],
        }

    # ── per-worker safety wrappers ─────────────────────────────────────

    async def _safe_search(self, competitor: dict[str, Any], research_questions: list[str]) -> dict[str, Any]:
        try:
            return await self.search_worker({
                "competitor": competitor,
                "research_questions": research_questions,
            })
        except Exception as exc:
            logger.warning("Search worker failed for %s: %s", competitor.get("name"), exc)
            return {}

    async def _safe_scrape(self, competitor: dict[str, Any], market_segment: str) -> dict[str, Any]:
        try:
            return await self.scrape_worker({
                "competitor": competitor,
                "market_segment": market_segment,
            })
        except Exception as exc:
            logger.warning("Scrape worker failed for %s: %s", competitor.get("name"), exc)
            return {}

    async def _safe_rag(self, competitor: dict[str, Any], market_segment: str) -> dict[str, Any]:
        try:
            return await self.rag_retriever({
                "competitor": competitor,
                "market_segment": market_segment,
            })
        except Exception as exc:
            logger.warning("RAG retriever failed for %s: %s", competitor.get("name"), exc)
            return {}

    # ── data processing (moved from graph.py) ───────────────────────────

    @staticmethod
    def _build_profile_summary(
        name: str,
        search: dict[str, Any],
        scrape: dict[str, Any],
        rag: dict[str, Any],
    ) -> str:
        """Build a data-derived profile summary from collected sources — no LLM call."""
        parts: list[str] = [name]

        search_results = search.get("search_results", []) or []
        for item in search_results[:3]:
            if isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("body") or item.get("snippet", "")).strip()
                if title:
                    parts.append(title)
                if snippet:
                    parts.append(snippet[:200])

        scraped = scrape.get("scraped_content", []) or []
        for item in scraped[:2]:
            if isinstance(item, dict):
                content = str(item.get("raw_content") or item.get("content", "")).strip()
                if content and len(content) > 20:
                    lines = [ln.strip() for ln in content.split("\n") if len(ln.strip()) > 15]
                    if lines:
                        parts.append("\n".join(lines[:4]))

        rag_context = rag.get("rag_context", []) or []
        for item in rag_context[:2]:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text and len(text) > 10:
                    if "Executive Summary:" in text or "Target: " in text:
                        continue
                    parts.append(text[:300])

        if len(parts) == 1:
            return f"{name} — insufficient data collected from search, scrape, or RAG."
        return " | ".join(parts)

    @staticmethod
    def _extract_sources(
        search: dict[str, Any],
        scrape: dict[str, Any],
        rag: dict[str, Any],
    ) -> list[str]:
        """Collect real URLs from all three data channels."""
        sources: list[str] = []
        seen: set[str] = set()
        for item in (search.get("search_results") or [])[:5]:
            url = str(item.get("href") or item.get("url", "")).strip()
            if url and url not in seen and url.startswith("http"):
                sources.append(url)
                seen.add(url)
        for item in (scrape.get("scraped_content") or [])[:3]:
            url = str(item.get("url") or item.get("source", "")).strip()
            if url and url not in seen and url.startswith("http"):
                sources.append(url)
                seen.add(url)
        for item in (rag.get("rag_context") or [])[:3]:
            url = str(item.get("source", "")).strip()
            if url and url not in seen and url.startswith("http"):
                sources.append(url)
                seen.add(url)
        return sources if sources else ["search_worker", "scrape_worker", "rag_retriever"]
