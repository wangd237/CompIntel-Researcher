"""Data-source registry for CompIntel Research.

Maps tool names to :class:`Tool` instances.  The profiler uses
:func:`get_enabled_tools` to discover which tools are available for a
given competitor, then calls them in parallel.

Adding a new data source (e.g. Crunchbase, PitchBook, a custom API)
requires exactly two steps:

1. Implement an async function with signature ``async def search(params:
   dict) -> dict``.
2. Add a ``Tool(...)`` entry to ``DATA_SOURCES`` below.

No changes to ``graph.py``, ``competitor_profiler.py``, or any other
pipeline component are needed.
"""

from __future__ import annotations

import logging
from typing import Any

from compintel.agents.rag_retriever import RAGRetriever
from compintel.agents.scrape_worker import ScrapeWorker
from compintel.agents.search_worker import SearchWorker

from . import Tool

logger = logging.getLogger(__name__)

# ── Tool adapters ──────────────────────────────────────────────────────
# Each adapter wraps an existing agent's __call__ into a plain async
# function that takes a dict of params.  This keeps the agent classes
# untouched — the Tool protocol is a thin veneer on top.


async def _search(params: dict[str, Any]) -> dict[str, Any]:
    """Adapter: SearchWorker.__call__ with dict params."""
    worker = _search_worker()
    return await worker(params)


async def _scrape(params: dict[str, Any]) -> dict[str, Any]:
    """Adapter: ScrapeWorker.__call__ with dict params."""
    worker = _scrape_worker()
    return await worker(params)


async def _rag_retrieve(params: dict[str, Any]) -> dict[str, Any]:
    """Adapter: RAGRetriever.__call__ with dict params."""
    worker = _rag_retriever()
    return await worker(params)


# ── Lazy worker singletons ─────────────────────────────────────────────

_workers: dict[str, Any] = {}


def _search_worker() -> SearchWorker:
    if "search" not in _workers:
        _workers["search"] = SearchWorker()
    return _workers["search"]


def _scrape_worker() -> ScrapeWorker:
    if "scrape" not in _workers:
        _workers["scrape"] = ScrapeWorker()
    return _workers["scrape"]


def _rag_retriever() -> RAGRetriever:
    if "rag" not in _workers:
        _workers["rag"] = RAGRetriever()
    return _workers["rag"]


# ── Enabled predicates ──────────────────────────────────────────────────


def _scrape_enabled(params: dict[str, Any]) -> bool:
    """Scraping requires a website URL or a market_segment to target."""
    competitor = params.get("competitor", {})
    if isinstance(competitor, dict) and competitor.get("website"):
        return True
    if params.get("market_segment"):
        return True
    return False


# ── Registry ───────────────────────────────────────────────────────────

DATA_SOURCES: dict[str, Tool] = {
    "web_search": Tool(
        name="web_search",
        description="Search the web for competitor news, analysis, and market data via Tavily or SerpAPI",
        parameters={
            "type": "object",
            "properties": {
                "competitor": {
                    "type": "object",
                    "description": "Competitor dict with name, website, etc.",
                },
                "research_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Research questions to build search queries from",
                },
            },
            "required": ["competitor"],
        },
        execute=_search,
        enabled=None,  # always on — has its own error/no-key fallback
    ),
    "web_scrape": Tool(
        name="web_scrape",
        description="Scrape competitor website and industry review sites for product, pricing, and positioning data",
        parameters={
            "type": "object",
            "properties": {
                "competitor": {
                    "type": "object",
                    "description": "Competitor dict with name and website URL",
                },
                "market_segment": {
                    "type": "string",
                    "description": "Market segment for industry-specific scrape targets",
                },
            },
            "required": ["competitor"],
        },
        execute=_scrape,
        enabled=_scrape_enabled,
    ),
    "rag_retrieve": Tool(
        name="rag_retrieve",
        description="Retrieve historical competitive analysis from the Qdrant self-growing memory",
        parameters={
            "type": "object",
            "properties": {
                "competitor": {
                    "type": "object",
                    "description": "Competitor dict with name",
                },
                "market_segment": {
                    "type": "string",
                    "description": "Market segment for cross-domain filtering",
                },
            },
            "required": ["competitor"],
        },
        execute=_rag_retrieve,
        enabled=None,  # always on — returns empty list when store is empty
    ),
}


def get_enabled_tools(params: dict[str, Any]) -> dict[str, Tool]:
    """Return tools whose ``enabled`` predicate passes for *params*.

    Tools with ``enabled=None`` are always included.
    """
    result: dict[str, Tool] = {}
    for name, tool in DATA_SOURCES.items():
        if tool.enabled is None or tool.enabled(params):
            result[name] = tool
    return result


def list_tools() -> list[dict[str, Any]]:
    """Return a human-readable tool listing (e.g. for /tools debug endpoint)."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in DATA_SOURCES.values()
    ]
