"""Search worker for competitor profiling."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from ..settings import CompIntelSettings
from .base import BaseCompIntelAgent


class SearchWorker(BaseCompIntelAgent):
    def __init__(
        self,
        model: str = "deepseek-chat",
        client: Any | None = None,
        settings: CompIntelSettings | None = None,
    ) -> None:
        super().__init__(model=model)
        self.client = client
        self.settings = settings

    async def __call__(self, state: Any) -> dict[str, Any]:
        competitor = {}
        questions: list[str] = []
        if isinstance(state, dict):
            competitor = state.get("competitor") or {}
            questions = state.get("research_questions") or []

        name = competitor.get("name", "unknown")
        queries = self._build_queries(name, questions)
        settings = self.settings or CompIntelSettings.from_env()

        if not settings.search_api_key and self.client is None:
            return self._error_response(
                queries,
                settings.search_provider,
                "Search API key is not configured.",
            )

        results: list[dict[str, Any]] = []
        for query in queries:
            try:
                provider_results = await self._search(query, settings)
            except Exception as exc:
                provider_results = [
                    {
                        "query": query,
                        "source": settings.search_provider,
                        "error": True,
                        "message": str(exc),
                    }
                ]
            results.extend(provider_results)

        results = self._dedupe_and_limit(results, limit=20)
        error_count = sum(1 for result in results if result.get("error"))
        return {
            "search_results": results,
            "execution_log": [
                {
                    "node": "search_worker",
                    "event": "completed",
                    "detail": f"{len(results)} results, {error_count} errors",
                }
            ],
        }

    def _build_queries(self, name: str, questions: list[str]) -> list[str]:
        if not questions:
            return [f"{name} competitive analysis"]
        return [
            f"{name} {question} competitive analysis"
            for question in questions
            if str(question).strip()
        ] or [f"{name} competitive analysis"]

    async def _search(self, query: str, settings: CompIntelSettings) -> list[dict[str, Any]]:
        provider = settings.search_provider.lower()
        if provider == "tavily":
            return await asyncio.to_thread(self._search_tavily, query, settings)
        if provider == "serpapi":
            return await asyncio.to_thread(self._search_serpapi, query, settings)
        return [
            {
                "query": query,
                "source": provider,
                "error": True,
                "message": f"Unsupported search provider: {provider}",
            }
        ]

    def _search_tavily(self, query: str, settings: CompIntelSettings) -> list[dict[str, Any]]:
        client = self.client
        if client is None:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.search_api_key)

        payload = client.search(query, search_depth="basic", max_results=5)
        return [
            self._normalize_result(item, query=query, source="tavily")
            for item in payload.get("results", [])
        ]

    def _search_serpapi(self, query: str, settings: CompIntelSettings) -> list[dict[str, Any]]:
        if self.client is not None:
            payload = self.client.search(query)
        else:
            params = urlencode(
                {
                    "engine": "google",
                    "q": query,
                    "api_key": settings.search_api_key,
                    "num": 5,
                }
            )
            with urlopen(f"https://serpapi.com/search.json?{params}", timeout=30) as response:
                import json

                payload = json.loads(response.read().decode("utf-8"))

        organic = payload.get("organic_results", [])
        return [
            self._normalize_result(item, query=query, source="serpapi")
            for item in organic
        ]

    def _normalize_result(self, item: dict[str, Any], query: str, source: str) -> dict[str, Any]:
        return {
            "title": item.get("title") or item.get("name") or "Untitled result",
            "url": item.get("url") or item.get("link"),
            "snippet": item.get("content") or item.get("snippet") or item.get("description") or "",
            "query": query,
            "source": source,
        }

    def _dedupe_and_limit(self, results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for result in results:
            url = str(result.get("url") or "").strip()
            if url:
                normalized_url = url.rstrip("/").lower()
                if normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
            deduped.append(result)
            if len(deduped) >= limit:
                break
        return deduped

    def _error_response(self, queries: list[str], provider: str, message: str) -> dict[str, Any]:
        results = [
            {
                "query": query,
                "source": provider,
                "error": True,
                "message": message,
            }
            for query in queries
        ]
        return {
            "search_results": results,
            "execution_log": [
                {
                    "node": "search_worker",
                    "event": "completed_with_error",
                    "detail": message,
                }
            ],
        }
