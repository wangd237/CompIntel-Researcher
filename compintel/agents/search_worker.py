"""Search worker for competitor profiling.

Performs web search with adaptive reformulation: if the first round of
results is low-quality (too few hits, mostly PR fluff, short snippets),
the worker rewrites the query to use different search angles and retries
up to 2 additional rounds, keeping the best-scoring batch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from ..settings import CompIntelSettings
from .base import BaseCompIntelAgent

# ── Windows SSL certificate workaround ──────────────────────────────────
# Python on Windows (especially in Conda environments) often cannot find
# the system CA bundle, causing SSLEOFError on HTTPS connections that
# work fine in curl / browsers.  This helper locates a valid bundle at
# module-load time and sets SSL_CERT_FILE if needed.

_WINDOWS_CA_PATHS = [
    # certifi (commonly installed alongside requests / httpx)
    lambda: __import__("certifi").where(),
    # Conda standard paths
    os.path.expandvars(r"%CONDA_PREFIX%\Library\ssl\cacert.pem"),
    os.path.expandvars(r"%CONDA_PREFIX%\Library\ssl\cert.pem"),
    os.path.expandvars(r"%CONDA_PREFIX%\ssl\cert.pem"),
    # pip / system Python
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python310\Lib\site-packages\pip\_vendor\certifi\cacert.pem"),
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\pip\_vendor\certifi\cacert.pem"),
]

def _ensure_ssl_certs() -> None:
    """Try to set SSL_CERT_FILE so that httpx/requests can validate TLS."""
    if os.environ.get("SSL_CERT_FILE"):
        return  # already configured

    if sys.platform != "win32":
        return  # macOS/Linux typically find certs via system paths

    for candidate in _WINDOWS_CA_PATHS:
        try:
            path = candidate() if callable(candidate) else candidate
        except Exception:
            continue
        if path and os.path.isfile(path):
            os.environ["SSL_CERT_FILE"] = path
            return


_ensure_ssl_certs()

logger = logging.getLogger(__name__)

# Query-quality evaluation thresholds
_MIN_RESULTS = 3
_MIN_MEANINGFUL_SNIPPET_FRACTION = 0.5
_MAX_PR_FRACTION = 0.7

# Keywords that indicate a result is a press release / PR fluff rather than
# substantive analysis — results with these in the title are down-weighted.
_PR_TITLE_MARKERS: tuple[str, ...] = (
    "press release", "announces", "announce", "announced",
    "unveils", "unveiled", "launches", "launched",
    "新闻稿", "发布", "宣布", "上线", "重磅",
    "raises $", "raises €", "funding round", "series ",
    "secures $", "closes $",
)

# Keywords that indicate substantive analysis content — results with these
# in the title are up-weighted.
_ANALYSIS_TITLE_MARKERS: tuple[str, ...] = (
    "analysis", "comparison", "vs", "versus", "competitive",
    "market share", "市场份额", "market position",
    "分析", "对比", "竞品", "竞争", "格局", "评测",
    "strategy", "战略", "review", "report",
)

# Suffix fragments appended during reformulation to steer toward richer results.
_ANALYSIS_SUFFIXES: list[str] = [
    "analysis comparison competitors market share",
    "vs competitors 竞争 对比 分析",
    "行业分析 市场份额 竞争格局",
]
_NEWS_SUFFIXES: list[str] = [
    "industry report analysis",
    "深度 行业报告 分析",
]


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
            best_for_query = await self._search_with_retry(query, settings)
            results.extend(best_for_query)

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

    # ── ReAct search loop ─────────────────────────────────────────────────

    async def _search_with_retry(
        self, query: str, settings: CompIntelSettings, max_rounds: int = 3,
    ) -> list[dict[str, Any]]:
        """Search → evaluate → reformulate loop.

        Keeps the best-scoring batch across all rounds.  Stops early when
        quality passes the threshold.
        """
        best_results: list[dict[str, Any]] = []
        best_score: float = -1.0
        current_query = query

        for attempt in range(max_rounds):
            try:
                batch = await self._search(current_query, settings)
            except Exception as exc:
                logger.warning("Search round %d failed for %r: %s", attempt + 1, current_query, exc)
                batch = [{"query": current_query, "source": settings.search_provider,
                          "error": True, "message": str(exc)}]

            score, detail = self._evaluate_quality(batch)

            if score > best_score:
                best_score = score
                best_results = batch

            if score >= 1.0:
                logger.debug("Search round %d passed (score=%.2f) for %r", attempt + 1, score, current_query)
                return best_results

            if attempt < max_rounds - 1:
                current_query = self._reformulate_query(query, attempt, detail)
                logger.debug("Search round %d reformulated %r → %r", attempt + 1, query, current_query)

        return best_results

    # ── quality evaluation ────────────────────────────────────────────────

    @staticmethod
    def _evaluate_quality(results: list[dict[str, Any]]) -> tuple[float, str]:
        """Score a batch of search results from 0.0 (useless) to 1.0+ (excellent).

        Returns (score, reason_tag) where reason_tag is one of:
        ``"too_few"``, ``"shallow"``, ``"pr_fluff"``, ``"ok"``, ``"good"``.
        """
        valid = [r for r in results if not r.get("error")]
        error_count = len(results) - len(valid)

        # ── count check ──
        if len(valid) < _MIN_RESULTS and error_count > 0:
            return 0.0, "too_few"
        if len(valid) == 0:
            return 0.0, "too_few"

        # ── snippet depth ──
        meaningful = sum(
            1 for r in valid
            if len(str(r.get("snippet", "")).strip()) >= 50
        )
        meaningful_frac = meaningful / max(1, len(valid))

        if meaningful_frac < _MIN_MEANINGFUL_SNIPPET_FRACTION:
            if meaningful == 0:
                return 0.1, "shallow"
            return 0.2, "shallow"

        # ── PR / news fluff ──
        pr_count = sum(
            1 for r in valid
            if any(marker in str(r.get("title", "")).lower() for marker in _PR_TITLE_MARKERS)
        )
        pr_frac = pr_count / max(1, len(valid))

        # ── analysis signals ──
        analysis_hits = sum(
            1 for r in valid
            if any(marker in str(r.get("title", "")).lower() for marker in _ANALYSIS_TITLE_MARKERS)
        )

        if pr_frac >= _MAX_PR_FRACTION and analysis_hits == 0:
            return 0.3, "pr_fluff"

        if analysis_hits >= 1 and pr_frac < 0.5:
            return 1.2, "good"

        return 1.0, "ok"

    # ── query reformulation ────────────────────────────────────────────────

    @staticmethod
    def _reformulate_query(original: str, attempt: int, reason_tag: str) -> str:
        """Rewrite *original* for a different search angle based on *reason_tag*.

        Each attempt uses a different strategy so that repeated failures
        explore complementary angles rather than trying the same thing
        again.
        """
        # Round 1 (attempt 0): append analysis keywords
        if attempt == 0:
            suffix = _ANALYSIS_SUFFIXES[0]
            if len(original) + len(suffix) + 1 < 200:
                return f"{original} {suffix}"
            return original

        # Round 2 (attempt 1): switch to a news / report angle
        suffix = _NEWS_SUFFIXES[attempt - 1] if attempt - 1 < len(_NEWS_SUFFIXES) else _NEWS_SUFFIXES[0]
        if len(original) + len(suffix) + 1 < 200:
            return f"{original} {suffix}"
        return original

    # ── helpers ──────────────────────────────────────────────────────────

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
