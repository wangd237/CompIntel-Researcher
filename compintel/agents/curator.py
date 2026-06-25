"""Content curator for CompIntel Research.

Sits between the fan-out competitor profiler and downstream analysis agents.
Evaluates source quality, deduplicates, and assesses S/O/W/T evidence density
so that downstream agents (SWOT, report_writer) know which profiles have
sufficient data and which need conservative handling.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

from ..parsing import safe_json_dumps
from ..prompts import load_prompt
from .base import BaseCompIntelAgent

# ── Rules-based junk filtering ────────────────────────────────────────────

# Patterns in page content that indicate a junk / error / irrelevant page.
_JUNK_CONTENT_PATTERNS: tuple[str, ...] = (
    "404", "page not found", "页面不存在", "抱歉，您访问的页面不存在",
    "captcha", "请完成安全验证", "verify you are human",
    "access denied", "forbidden", "403", "登录后才能",
    "sign in to continue", "please enable javascript",
)

# URL patterns that are clearly not competitive intelligence sources.
_IRRELEVANT_URL_PATTERNS: tuple[str, ...] = (
    "facebook.com", "twitter.com", "instagram.com", "youtube.com",
    "linkedin.com/company", "linkedin.com/jobs",
    "amazon.com/shop", "shop.",
    "google.com/maps", "maps.google.",
)


def _is_junk_page(content: str) -> bool:
    """Return True if page content looks like junk / error / irrelevant."""
    lowered = content.lower()[:500]
    return any(pat in lowered for pat in _JUNK_CONTENT_PATTERNS)


def _is_relevant_url(url: str) -> bool:
    """Return False for URLs that are clearly not competitive intelligence."""
    lowered = url.lower()
    return not any(pat in lowered for pat in _IRRELEVANT_URL_PATTERNS)


def _deduplicate_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove search results that are near-duplicates by full URL path.

    Uses the complete URL (minus trailing slash and query string) as the
    dedup key — not just the domain.  This preserves multiple distinct
    pages from the same review site (e.g. G2, Capterra) where every
    result shares a domain but points to a different competitor page.
    """
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url", "")).strip()
        if not url:
            deduped.append(item)
            continue
        # Normalize: strip trailing slash + query/fragment for dedup key
        dedup_key = url.rstrip("/").lower()
        if "?" in dedup_key:
            dedup_key = dedup_key.split("?")[0]
        if dedup_key not in seen_urls:
            seen_urls.add(dedup_key)
            deduped.append(item)
    return deduped


def _count_evidence_sources(profile: dict[str, Any]) -> dict[str, int]:
    """Count how many data channels have real content for this profile."""
    counts = {"search": 0, "scrape": 0, "rag": 0, "total": 0}

    search_results = profile.get("search_results") or []
    for item in (search_results if isinstance(search_results, list) else []):
        snippet = str(item.get("snippet") or item.get("body", "")).strip()
        if len(snippet) > 20:
            counts["search"] += 1

    scraped = profile.get("scraped_content") or []
    for item in (scraped if isinstance(scraped, list) else []):
        content = str(item.get("content") or item.get("raw_content", "")).strip()
        if len(content) > 30 and not _is_junk_page(content):
            counts["scrape"] += 1

    rag = profile.get("rag_context") or []
    for item in (rag if isinstance(rag, list) else []):
        text = str(item.get("text", "")).strip()
        if len(text) > 20:
            counts["rag"] += 1

    counts["total"] = counts["search"] + counts["scrape"] + counts["rag"]
    return counts


# Labels we use to communicate data quality to downstream agents.
_DATA_QUALITY_LABELS = {
    # >= 2 search + >= 1 scrape → good coverage
    "rich": "RICH_DATA — 数据充足。strength/weakness/opportunity/threat 四个象限均应提供 2-3 条有 evidence 的条目。",
    # >= 1 search + any scrape or rag
    "adequate": "ADEQUATE_DATA — 数据尚可。S/O 象限提供 2 条有 evidence 的条目，W/T 象限如实标注数据不足即可。",
    # Some search but no real scrape/rag content
    "thin": "THIN_DATA — 数据偏薄。仅能支撑 1-2 条 S 或 O，W/T 标注 insufficient data 不要编造。",
    # No real data at all
    "empty": "EMPTY_DATA — 无有效数据。不要为这个竞品编造任何 SWOT 条目，output 中注明 data unavailable。",
}


class CuratorAgent(BaseCompIntelAgent):
    """Content curator: clean profiles, deduplicate sources, grade evidence quality."""

    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model, model_key="fast")
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        profiles = s.profiles
        market_segment = s.market_segment

        curated_profiles: list[dict[str, Any]] = []
        evidence_grades: list[dict[str, Any]] = []
        warnings: list[str] = []

        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            name = profile.get("name", "unknown")

            # Step 1: Rules-based cleanup
            cleaned = self._clean_profile(profile)

            # Step 2: Evidence assessment (deterministic, no LLM needed)
            evidence = _count_evidence_sources(cleaned)
            grade = self._grade_evidence(evidence)
            cleaned["evidence_grade"] = grade
            cleaned["evidence_counts"] = evidence

            evidence_grades.append({
                "name": name,
                "grade": grade,
                "counts": evidence,
            })

            # Step 3: Optional LLM-based quality summary (lightweight)
            if self.completion_fn is not None or self.llm.settings.llm_api_key:
                quality_note = await self._assess_quality_llm(name, cleaned, market_segment)
                if quality_note:
                    cleaned["curator_note"] = quality_note

            curated_profiles.append(cleaned)

            if evidence["total"] == 0:
                warnings.append(f"{name}: no real data collected — profile will be minimal")

        # Generate a market-level evidence summary for downstream agents
        evidence_summary = self._build_evidence_summary(evidence_grades)

        return {
            "curated_profiles": curated_profiles,
            "curator_evidence": evidence_summary,
            "warnings": warnings,
            "execution_log": [
                {"node": "curator", "event": "completed",
                 "detail": f"Curated {len(curated_profiles)} profiles: "
                           f"{evidence_summary.get('overall_grade', 'unknown')}"}
            ],
        }

    def _clean_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Rules-based cleanup: remove junk, deduplicate, truncate noise."""
        cleaned = dict(profile)

        # Clean search results
        search_results = profile.get("search_results") or []
        if isinstance(search_results, list):
            # Filter out junk URLs
            filtered = [
                item for item in search_results
                if isinstance(item, dict) and _is_relevant_url(
                    str(item.get("href") or item.get("url", ""))
                )
            ]
            # Deduplicate by domain
            cleaned["search_results"] = _deduplicate_search_results(filtered)

        # Clean scraped content
        scraped = profile.get("scraped_content") or []
        if isinstance(scraped, list):
            cleaned["scraped_content"] = [
                item for item in scraped
                if isinstance(item, dict)
                and not _is_junk_page(str(item.get("content") or item.get("raw_content", "")))
                and _is_relevant_url(str(item.get("url", "")))
            ]

        return cleaned

    @staticmethod
    def _grade_evidence(evidence: dict[str, int]) -> str:
        total = evidence["total"]
        search = evidence["search"]
        scrape = evidence["scrape"]
        rag = evidence["rag"]
        # Rich: at least two channels with substantial data
        if total >= 4 and (search >= 2 or scrape >= 2 or rag >= 1):
            return "rich"
        if total >= 2:
            return "adequate"
        if total >= 1:
            return "thin"
        return "empty"

    def _build_evidence_summary(self, grades: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a human-readable evidence summary for downstream agents."""
        grade_counts = {"rich": 0, "adequate": 0, "thin": 0, "empty": 0}
        for g in grades:
            grade_counts[g["grade"]] = grade_counts.get(g["grade"], 0) + 1

        total = len(grades)
        rich_pct = grade_counts["rich"] / max(total, 1) * 100

        if rich_pct >= 60:
            overall = "good"
        elif rich_pct >= 30:
            overall = "mixed"
        else:
            overall = "weak"

        per_competitor = {}
        for g in grades:
            name = g["name"]
            grade = g["grade"]
            per_competitor[name] = _DATA_QUALITY_LABELS.get(grade, "")

        return {
            "overall_grade": overall,
            "grade_distribution": grade_counts,
            "per_competitor": per_competitor,
        }

    async def _assess_quality_llm(
        self, name: str, profile: dict[str, Any], market_segment: str
    ) -> str | None:
        """Lightweight LLM call to summarize what this profile contains / lacks."""
        if self.completion_fn is not None:
            return None  # skip LLM for test-injected paths

        snippets: list[str] = []
        for item in (profile.get("search_results") or [])[:3]:
            if isinstance(item, dict):
                t = str(item.get("title", "")).strip()
                s = str(item.get("snippet") or item.get("body", "")).strip()[:120]
                if s:
                    snippets.append(f"{t}: {s}")

        if not snippets:
            return None

        prompt = load_prompt("curator")
        try:
            raw = await self.llm.call(
                prompt.format(
                    name=name,
                    market_segment=market_segment,
                    snippets="\n".join(snippets),
                ),
                model_key=prompt.model_key,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
            )
            return str(raw).strip()
        except Exception:
            return None
