"""Scrape worker for competitor profiling."""

from __future__ import annotations

import asyncio
import random
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from .base import BaseCompIntelAgent


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
]

# P1-2: Industry-specific review/comparison sites.
# Keys are Chinese/English market segment keywords (matched case-insensitive).
# Values are lists of URL templates where {query} is replaced by the URL-encoded
# competitor name.
_INDUSTRY_SCRAPE_SOURCES: dict[str, list[str]] = {
    # Automotive / NEV
    "汽车": [
        "https://www.autohome.com.cn/search?q={query}",
        "https://www.dongchedi.com/search?keyword={query}",
    ],
    "新能源": [
        "https://www.autohome.com.cn/search?q={query}",
        "https://www.dongchedi.com/search?keyword={query}",
    ],
    "nev": [
        "https://insideevs.com/search/{query}/",
        "https://cleantechnica.com/?s={query}",
    ],
    "ev": [
        "https://insideevs.com/search/{query}/",
    ],
    "automotive": [
        "https://www.caranddriver.com/search/?q={query}",
    ],
    # Software / SaaS (keep existing G2/Capterra as primary)
    "saas": [
        "https://www.g2.com/search?query={query}",
        "https://www.capterra.com/search/?query={query}",
    ],
    "软件": [
        "https://www.g2.com/search?query={query}",
        "https://www.capterra.com/search/?query={query}",
    ],
    "collaboration": [
        "https://www.g2.com/search?query={query}",
        "https://www.capterra.com/search/?query={query}",
    ],
    # Finance / investment
    "投资": [
        "https://www.google.com/search?q={query}+investment+portfolio",
        "https://crunchbase.com/organization/{query}",
    ],
    "investment": [
        "https://crunchbase.com/organization/{query}",
    ],
    # General fallback — search engine
    "_default": [
        "https://www.g2.com/search?query={query}",
        "https://www.capterra.com/search/?query={query}",
    ],
}

def _resolve_industry_sources(market_segment: str) -> list[str]:
    """Return scrape source URL templates that match *market_segment*."""
    if not market_segment:
        return _INDUSTRY_SCRAPE_SOURCES["_default"]
    lowered = market_segment.lower()
    for keyword, templates in _INDUSTRY_SCRAPE_SOURCES.items():
        if keyword == "_default":
            continue
        if keyword.lower() in lowered:
            return templates
    return _INDUSTRY_SCRAPE_SOURCES["_default"]


class BeautifulSoupScraper:
    def scrape(self, url: str, user_agent: str, timeout: float = 20) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            # Let BeautifulSoup detect encoding from <meta charset> /
            # Content-Type header.  decode("utf-8", errors="ignore")
            # silently drops GBK/GB2312 content from many Chinese sites.
            soup = BeautifulSoup(raw, "html.parser", from_encoding=response.headers.get_content_charset())

        for node in soup(["script", "style", "noscript"]):
            node.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        content = soup.get_text("\n", strip=True)
        return {
            "url": url,
            "title": title,
            "content": content,
        }


class ScrapeWorker(BaseCompIntelAgent):
    def __init__(
        self,
        model: str = "deepseek-chat",
        scraper: BeautifulSoupScraper | None = None,
        max_concurrency: int = 3,
        max_chars: int = 8000,
        min_delay: float = 0.1,
        max_delay: float = 0.4,
    ) -> None:
        super().__init__(model=model)
        self.scraper = scraper or BeautifulSoupScraper()
        self.max_concurrency = max_concurrency
        self.max_chars = max_chars
        self.min_delay = min_delay
        self.max_delay = max_delay

    async def __call__(self, state: Any) -> dict[str, Any]:
        competitor = {}
        market_segment = ""
        if isinstance(state, dict):
            competitor = state.get("competitor") or {}
            market_segment = str(state.get("market_segment", ""))

        name = competitor.get("name", "unknown")
        urls = self._build_target_urls(name, competitor.get("website"), market_segment)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        scraped = await asyncio.gather(
            *(self._scrape_one(url, semaphore) for url in urls)
        )
        error_count = sum(1 for item in scraped if item.get("error"))
        return {
            "scraped_content": scraped,
            "execution_log": [
                {
                    "node": "scrape_worker",
                    "event": "completed",
                    "detail": f"{len(scraped)} pages attempted, {error_count} errors",
                }
            ],
        }

    def _build_target_urls(self, name: str, website: str | None, market_segment: str = "") -> list[str]:
        urls: list[str] = []
        # Always scrape the competitor's own website
        if website:
            base_url = self._normalize_website(website)
            urls.extend(
                [
                    base_url,
                    urljoin(base_url.rstrip("/") + "/", "pricing"),
                    urljoin(base_url.rstrip("/") + "/", "about"),
                ]
            )

        # P1-2: Industry-specific review/comparison sites
        query = quote_plus(name)
        templates = _resolve_industry_sources(market_segment)
        for template in templates:
            urls.append(template.format(query=query))

        return self._dedupe_urls(urls)

    def _normalize_website(self, website: str) -> str:
        parsed = urlparse(website)
        if parsed.scheme:
            return website
        return f"https://{website}"

    def _dedupe_urls(self, urls: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for url in urls:
            normalized = url.rstrip("/")
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    async def _scrape_one(self, url: str, semaphore: asyncio.Semaphore) -> dict[str, Any]:
        async with semaphore:
            await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
            user_agent = random.choice(USER_AGENTS)
            try:
                result = await asyncio.to_thread(self.scraper.scrape, url, user_agent)
                content = str(result.get("content", ""))
                return {
                    "url": result.get("url") or url,
                    "title": result.get("title") or "",
                    "content": content[: self.max_chars],
                    "truncated": len(content) > self.max_chars,
                    "source": "beautifulsoup",
                }
            except Exception as exc:
                return {
                    "url": url,
                    "title": "",
                    "content": "",
                    "source": "beautifulsoup",
                    "error": True,
                    "message": str(exc),
                }
