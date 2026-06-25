"""Competitor profiling orchestration."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseCompIntelAgent
from .rag_retriever import RAGRetriever
from .scrape_worker import ScrapeWorker
from .search_worker import SearchWorker

logger = logging.getLogger(__name__)


class CompetitorProfilerAgent(BaseCompIntelAgent):
    """Fan-out parallel Search, Scrape, and RAG retrieval for each competitor via LangGraph subgraph."""

    def __init__(self, model: str = "deepseek-chat") -> None:
        super().__init__(model=model)
        self.search_worker = SearchWorker(model=model)
        self.scrape_worker = ScrapeWorker(model=model)
        self.rag_retriever = RAGRetriever(model=model)

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        competitor = s.competitor
        research_questions = s.research_questions

        try:
            search = await self.search_worker({"competitor": competitor, "research_questions": research_questions})
        except Exception as exc:
            logger.warning("Search worker failed for %s: %s", competitor.get("name"), exc)
            search = {}
        try:
            scrape = await self.scrape_worker({"competitor": competitor})
        except Exception as exc:
            logger.warning("Scrape worker failed for %s: %s", competitor.get("name"), exc)
            scrape = {}
        try:
            rag = await self.rag_retriever({"competitor": competitor})
        except Exception as exc:
            logger.warning("RAG retriever failed for %s: %s", competitor.get("name"), exc)
            rag = {}

        profile = {
            "name": competitor.get("name", "unknown"),
            "website": competitor.get("website"),
            "summary": f"Profile summary for {competitor.get('name', 'unknown')}.",
            "search_results": search.get("search_results", []),
            "scraped_content": scrape.get("scraped_content", []),
            "rag_context": rag.get("rag_context", []),
            "sources": [
                "search_worker",
                "scrape_worker",
                "rag_retriever",
            ],
        }
        return {
            "profile": profile,
            "profiles": [profile],
            "execution_log": [
                {"node": "competitor_profiler", "event": "completed", "detail": profile["name"]}
            ],
        }
