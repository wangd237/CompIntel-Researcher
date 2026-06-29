"""
Intent analyst for CompIntel Research.

Primary path: LLM parsing with retries and output validation.
Heuristic path: only when no API key is configured — never as a
fallback from LLM timeout (retries handle that).
"""

from __future__ import annotations
import asyncio
import logging

import re
from typing import Any

logger = logging.getLogger(__name__)

from ..parsing import load_repaired_json
from ..prompts import load_prompt
from ..settings import CompIntelSettings
from ..schemas import CompetitorCandidate, IntentAnalysisResponse
from .base import BaseCompIntelAgent

# Chinese enumeration comma (U+3001) and English comma/semicolon — used to
# split user-provided company names out of a query on the heuristic path.
_NAME_SPLIT_RE = re.compile(r"[、,，;；\s]+")

# Generic English keywords that are NEVER valid competitor names.
# These are dimension labels (pricing, market, technology, product) or
# query artifacts that the noise filter must catch in both LLM and
# heuristic outputs.
_INVALID_COMPETITOR_NAMES: set[str] = {
    "pricing", "market", "technology", "product", "competitor",
    "competitors", "analysis", "overview", "summary", "report",
    "research", "features", "strategy", "company", "companies",
    "unknown", "placeholder", "n/a", "none", "test", "example",
    "分析", "研究", "对比", "竞品", "竞争", "市场", "格局", "主要竞品",
}

_QUERY_ARTIFACT_FRAGMENTS: tuple[str, ...] = (
    "分析", "研究", "对比", "竞品", "竞争格局", "市场", "赛道", "领域",
    "主要", "工具", "行业", "报告",
)

_KNOWN_COMPETITOR_SEEDS: dict[str, list[tuple[str, str | None]]] = {
    "notion": [
        ("Coda", "https://coda.io"),
        ("Confluence", "https://www.atlassian.com/software/confluence"),
        ("Microsoft Loop", "https://www.microsoft.com/microsoft-loop"),
    ],
    "slack": [
        ("Microsoft Teams", "https://www.microsoft.com/microsoft-teams/group-chat-software"),
        ("Discord", "https://discord.com"),
        ("Google Chat", "https://workspace.google.com/products/chat/"),
    ],
    "figma": [
        ("Adobe XD", None),
        ("Sketch", "https://www.sketch.com"),
        ("Canva", "https://www.canva.com"),
    ],
}


def _split_names(raw: str) -> list[str]:
    """Split a user-provided name list on Chinese / English separators."""
    parts = _NAME_SPLIT_RE.split(raw.strip())
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 2]


def _is_plausible_competitor(name: str) -> bool:
    """Return False for names that are clearly not company names."""
    clean = name.strip(" ：:，,。.、 ")
    if not clean or len(clean) < 2:
        return False
    lowered = clean.lower()
    if lowered in _INVALID_COMPETITOR_NAMES:
        return False
    if any(fragment in clean for fragment in _QUERY_ARTIFACT_FRAGMENTS) and not re.search(r"[A-Za-z]", clean):
        return False
    # Single Chinese character is not a company name
    if len(clean) == 1 and '一' <= clean <= '鿿':
        return False
    return True


class IntentAnalystAgent(BaseCompIntelAgent):
    """Intent parser with LLM-primary + heuristic-fallback paths."""

    async def __call__(self, state: Any) -> dict[str, Any]:
        settings = CompIntelSettings.from_env()
        # Support re-parsing a raw LLM response
        if isinstance(state, dict) and "raw_response" in state:
            parsed = load_repaired_json(str(state["raw_response"]))
            if isinstance(parsed, dict):
                return self._normalize_payload(parsed)

        # Handle both string (direct call) and dict (graph invocation)
        if isinstance(state, str):
            query = state
        else:
            query = self.read_state(state).query

        # LLM path with retries — only skip when no API key is configured
        if settings.openai_api_key:
            llm_payload = await self._try_llm_parse(query, settings)
            if llm_payload is not None:
                return self._normalize_payload(llm_payload)

        # Heuristic path (no API key or all LLM retries exhausted)
        target = self._extract_target(query)
        competitors = self._seed_competitors(target, query)
        result = IntentAnalysisResponse(
            target=target,
            market_segment=self._infer_segment(query, target),
            competitors=competitors,
            research_questions=self._build_questions(target, query),
            notes=[
                "Heuristic fallback used — LLM unavailable or all retries exhausted. "
                "Competitor names may be incomplete.",
            ],
        )
        return self._normalize_payload({
            "intent": result.model_dump(),
            "target": result.target,
            "market_segment": result.market_segment,
            "competitors": [competitor.model_dump() for competitor in competitors],
            "research_questions": result.research_questions,
            "notes": result.notes,
        })

    async def _try_llm_parse(self, query: str, settings: CompIntelSettings) -> dict[str, Any] | None:
        """Call the LLM via LLMService and validate the output."""
        if not settings.openai_api_key:
            return None

        prompt = load_prompt("intent_analyst")
        parsed = await self.llm.call_and_parse(
            prompt.format(query=query),
            model_key=prompt.model_key,
            max_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
            timeout=45.0,
        )
        if not isinstance(parsed, dict):
            logger.warning("Intent LLM returned unparseable JSON after retries")
            return None

        # Validate the output
        competitors = parsed.get("competitors", [])
        valid_competitors = [
            c for c in competitors
            if isinstance(c, dict) and _is_plausible_competitor(c.get("name", ""))
        ]
        rejected = len(competitors) - len(valid_competitors)
        if rejected:
            logger.info(
                "Intent validator rejected %d non-company name(s): %s",
                rejected,
                [c.get("name") for c in competitors if isinstance(c, dict) and not _is_plausible_competitor(c.get("name", ""))],
            )

        parsed["competitors"] = valid_competitors
        return parsed

        return None

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        intent = payload.get("intent") or {}
        competitors = payload.get("competitors") or intent.get("competitors") or []
        questions = payload.get("research_questions") or intent.get("research_questions") or []
        notes = payload.get("notes") or intent.get("notes") or []
        target = payload.get("target") or intent.get("target") or "unknown"
        market_segment = payload.get("market_segment") or intent.get("market_segment") or "unknown"

        # Cap competitors at 5 (hard limit) to prevent fan-out explosion and
        # keep the comparison matrix scannable.  LLM + heuristic + seed paths
        # all funnel through here so this is the single enforcement point.
        MAX_COMPETITORS = 5
        if len(competitors) > MAX_COMPETITORS:
            truncated_names = [
                (c.get("name") if isinstance(c, dict) else str(c))
                for c in competitors[MAX_COMPETITORS:]
            ]
            notes.append(
                f"竞品数量从 {len(competitors)} 截断至 {MAX_COMPETITORS}，"
                f"已移除: {', '.join(truncated_names)}"
            )
            logger.info("Truncated competitors from %d to %d: removed %s",
                        len(competitors), MAX_COMPETITORS, truncated_names)
            competitors = competitors[:MAX_COMPETITORS]

        normalized_intent = {
            **intent,
            "target": target,
            "market_segment": market_segment,
            "competitors": [
                competitor.model_dump() if isinstance(competitor, CompetitorCandidate) else competitor
                for competitor in competitors
            ],
            "research_questions": [str(item).strip() for item in questions if str(item).strip()],
            "notes": [str(item).strip() for item in notes if str(item).strip()],
        }
        return {
            "intent": normalized_intent,
            "target": target,
            "market_segment": market_segment,
            "competitors": normalized_intent["competitors"],
            "research_questions": normalized_intent["research_questions"],
            "notes": normalized_intent["notes"],
        }

    def _extract_target(self, query: str) -> str:
        """Extract the primary analysis target from the query.

        When the user writes e.g. "分析 Notion、Coda 的竞品格局" the first
        company is treated as the primary *target* and the remainder as
        *seed competitors* (see ``_seed_competitors``).
        """
        patterns = [
            r"(?:分析|研究|对比)\s*([A-Za-z0-9一-鿿 .&\-—–、]+?)(?:的竞品|竞争格局|在|市场|$)",
            r"(?:what is|who are|analyze|research)\s+([A-Za-z0-9 .&\-—–]+?)\s+(?:competitors|competition|competitive)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                raw = match.group(1).strip(" ：:，,。.、 ")
                # If raw still contains 、, take only the first fragment.
                parts = _split_names(raw)
                return parts[0] if parts else (raw or "unknown")

        cleaned = query.strip()
        # Last resort: try to salvage the first named entity from the query.
        names = _split_names(cleaned)
        if names:
            return names[0]
        return cleaned[:48] if cleaned else "unknown"

    def _infer_segment(self, query: str, target: str) -> str:
        if "协作" in query or "collabor" in query.lower():
            return "collaboration software"
        if "笔记" in query or "note" in query.lower():
            return "knowledge management"
        return f"{target} related market"

    def _seed_competitors(self, target: str, query: str = "") -> list[CompetitorCandidate]:
        """Return user-named competitors extracted from the query.

        When the user writes e.g. "分析 Notion、Coda 的竞品格局", the heuristic
        extracts the target (Notion) and returns the remaining names (Coda) as
        seed competitors.  Dimension keywords (pricing, market, technology, ...)
        are filtered out.
        """
        competitors = self._extract_explicit_competitors(target, query)

        # If the query mentions a competitor via "vs X" / "与 X 对比"
        explicit_mention = re.search(
            r"(?:与|和|vs\.?|versus|against|compete[sd]?\s+(?:with|against))\s*([A-Z][A-Za-z0-9_\-.]+)",
            query, flags=re.IGNORECASE,
        )
        if explicit_mention:
            name = explicit_mention.group(1).strip()
            existing_names = {candidate[0] for candidate in competitors}
            if name != target and name not in existing_names and _is_plausible_competitor(name):
                competitors.append((name, None, "explicit"))

        if not competitors:
            competitors.extend(self._known_seed_competitors(target))

        return [
            CompetitorCandidate(
                name=name,
                website=website,
                rationale="用户明确输入" if source == "explicit" else "启发式种子竞品",
            )
            for name, website, source in competitors
        ]

    def _extract_explicit_competitors(self, target: str, query: str) -> list[tuple[str, str | None, str]]:
        names: list[tuple[str, str | None, str]] = []
        if "、" not in query and "," not in query and "，" not in query:
            return names

        target_index = query.lower().find(target.lower())
        if target_index == -1:
            return names
        tail = query[target_index + len(target):]
        tail = re.split(r"(?:的竞品|竞争格局|在|市场|行业|赛道)", tail, maxsplit=1)[0]
        for name in _split_names(tail):
            clean = name.strip(" ：:，,。.、 ")
            if clean and clean != target and _is_plausible_competitor(clean):
                names.append((clean, None, "explicit"))
        return names

    def _known_seed_competitors(self, target: str) -> list[tuple[str, str | None, str]]:
        seeds = _KNOWN_COMPETITOR_SEEDS.get(target.lower(), [])
        return [(name, website, "seed") for name, website in seeds]

    def _build_questions(self, target: str, query: str = "") -> list[str]:
        """Generate broader search questions when LLM is unavailable.

        Instead of fabricating fake competitor names (which pollutes downstream),
        use industry keywords and the raw query to help SearchWorker find real
        competitors organically.
        """
        parts: list[str] = [target]
        if query:
            # extract key industry terms from the original query
            keywords = ["投资", "竞争", "investment", "competitor", "market",
                        "technology", "AI", "strategy", "portfolio"]
            for kw in keywords:
                if kw.lower() in query.lower():
                    parts.append(kw)
        topic = " ".join(dict.fromkeys(parts))  # deduplicate while preserving order
        return [
            f"{topic} 主要竞争对手及市场格局",
            f"{topic} 近年投资策略及重点领域",
            f"{topic} 与同赛道玩家的对比分析",
        ]
