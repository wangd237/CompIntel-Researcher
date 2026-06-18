"""Intent analyst for CompIntel Research."""

from __future__ import annotations
import logging

import re
from typing import Any

logger = logging.getLogger(__name__)

from ..llm import _split_provider_model
from ..parsing import load_repaired_json
from ..settings import CompIntelSettings
from ..schemas import CompetitorCandidate, IntentAnalysisResponse
from .base import BaseCompIntelAgent


class IntentAnalystAgent(BaseCompIntelAgent):
    """Heuristic first-pass intent parser.

    The production version will call an LLM, but this gives us a runnable
    Week 1 scaffold and a clear contract for the rest of the graph.
    """

    async def __call__(self, state: Any) -> dict[str, Any]:
        settings = CompIntelSettings.from_env()
        if isinstance(state, dict) and "raw_response" in state:
            parsed = load_repaired_json(str(state["raw_response"]))
            if isinstance(parsed, dict):
                return self._normalize_payload(parsed)

        query = state if isinstance(state, str) else str(getattr(state, "get", lambda *_: "")("query", ""))
        llm_payload = await self._try_llm_parse(query, settings)
        if llm_payload is not None:
            return self._normalize_payload(llm_payload)

        target = self._extract_target(query)
        competitors = self._seed_competitors(target)
        result = IntentAnalysisResponse(
            target=target,
            market_segment=self._infer_segment(query, target),
            competitors=competitors,
            research_questions=self._build_questions(target),
            notes=[
                "Heuristic fallback used because no LLM adapter is wired yet.",
                "Replace with JSON-validated model call in Week 1.",
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
        if not settings.openai_api_key:
            return None

        try:
            from ..llm import create_chat_completion
        except Exception:
            logger.exception("Failed to import create_chat_completion")
            return None

        provider, model = _split_provider_model(settings.fast_llm)

        prompt = (
            "You are CompIntel's intent analyst.\n"
            "Extract target, market_segment, competitors, research_questions, and notes.\n"
            "Return strict JSON only.\n"
            f"Query: {query}\n"
            "JSON schema:\n"
            "{"
            '"target": string, '
            '"market_segment": string, '
            '"competitors": [{"name": string, "website": string|null, "rationale": string|null}], '
            '"research_questions": [string], '
            '"notes": [string]'
            "}"
        )
        try:
            raw = await create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=1000,
                temperature=0.2,
            )
        except Exception:
            logger.exception("LLM call failed, returning None")
            return None
        parsed = load_repaired_json(raw)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        intent = payload.get("intent") or {}
        competitors = payload.get("competitors") or intent.get("competitors") or []
        questions = payload.get("research_questions") or intent.get("research_questions") or []
        notes = payload.get("notes") or intent.get("notes") or []
        target = payload.get("target") or intent.get("target") or "unknown"
        market_segment = payload.get("market_segment") or intent.get("market_segment") or "unknown"
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
        patterns = [
            r"(?:分析|研究|对比)\s*([A-Za-z0-9\u4e00-\u9fff .&\-]+?)(?:的竞品|竞争格局|在|市场|$)",
            r"(?:what is|who are|analyze|research)\s+([A-Za-z0-9 .&\-]+?)\s+(?:competitors|competition|competitive)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" ：:，,。. ")
        cleaned = query.strip()
        return cleaned[:48] if cleaned else "unknown"

    def _infer_segment(self, query: str, target: str) -> str:
        if "协作" in query or "collabor" in query.lower():
            return "collaboration software"
        if "笔记" in query or "note" in query.lower():
            return "knowledge management"
        return f"{target} related market"

    def _seed_competitors(self, target: str) -> list[CompetitorCandidate]:
        base = target or "target"
        seeds = [
            CompetitorCandidate(name=f"{base} Alternative A", rationale="Likely direct substitute"),
            CompetitorCandidate(name=f"{base} Alternative B", rationale="Likely platform competitor"),
        ]
        return seeds

    def _build_questions(self, target: str) -> list[str]:
        return [
            f"{target} 的核心产品定位是什么？",
            f"{target} 的主要竞品有哪些？",
            f"{target} 的定价、分发和生态位如何？",
            f"{target} 相比竞品的差异化优势是什么？",
        ]
