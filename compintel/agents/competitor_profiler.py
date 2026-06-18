"""Competitor profiling orchestration."""

from __future__ import annotations

from typing import Any

from .base import BaseCompIntelAgent
from .rag_retriever import RAGRetriever
from .scrape_worker import ScrapeWorker
from .search_worker import SearchWorker


class CompetitorProfilerAgent(BaseCompIntelAgent):
    """Fan-out parallel Search, Scrape, and RAG retrieval for each competitor via LangGraph subgraph."""

    def __init__(self, model: str = "deepseek-chat") -> None:
        super().__init__(model=model)
        self.search_worker = SearchWorker(model=model)
        self.scrape_worker = ScrapeWorker(model=model)
        self.rag_retriever = RAGRetriever(model=model)

    async def __call__(self, state: Any) -> dict[str, Any]:
        competitor = {}
        research_questions: list[str] = []
        if isinstance(state, dict):
            competitor = state.get("competitor") or {}
            research_questions = state.get("research_questions") or []

        search = await self.search_worker({"competitor": competitor, "research_questions": research_questions})
        scrape = await self.scrape_worker({"competitor": competitor})
        rag = await self.rag_retriever({"competitor": competitor})

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
