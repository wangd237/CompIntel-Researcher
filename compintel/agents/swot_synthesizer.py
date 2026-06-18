"""SWOT synthesis for CompIntel Research."""

from __future__ import annotations
import logging

from typing import Any

logger = logging.getLogger(__name__)

from ..llm import _split_provider_model
from ..parsing import load_repaired_json, safe_json_dumps
from ..settings import CompIntelSettings
from .base import BaseCompIntelAgent


class SWOTSynthesizerAgent(BaseCompIntelAgent):
    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model)
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        profiles = []
        market_analysis = {}
        if isinstance(state, dict):
            profiles = state.get("profiles") or []
            market_analysis = state.get("market_analysis") or {}

        settings = CompIntelSettings.from_env()
        swot = await self._try_llm_synthesize(profiles, market_analysis, settings)
        source = "llm"
        if swot is None:
            if settings.llm_api_key:
                swot = self._derived_swot(profiles, market_analysis)
                source = "derived"
            else:
                swot = self._fallback_swot(profiles, market_analysis)
                source = "template"

        return {
            "swot_analysis": swot,
            "execution_log": [
                {"node": "swot_synthesizer", "event": "completed", "detail": f"{source}: swot synthesized"}
            ],
        }

    async def _try_llm_synthesize(
        self,
        profiles: list[dict[str, Any]],
        market_analysis: dict[str, Any],
        settings: CompIntelSettings,
    ) -> dict[str, Any] | None:
        if not settings.llm_api_key and self.completion_fn is None:
            return None

        completion_fn = self.completion_fn
        if completion_fn is None:
            try:
                from ..llm import create_chat_completion
            except Exception:
                logger.exception("Failed to import create_chat_completion")
                return None
            completion_fn = create_chat_completion

        provider, model = _split_provider_model(settings.strategic_llm)
        compact_profiles = [
            {
                "name": profile.get("name"),
                "summary": profile.get("summary"),
                "sources": profile.get("sources", []),
                "search_results": profile.get("search_results", [])[:5],
                "scraped_content": profile.get("scraped_content", [])[:3],
                "rag_context": profile.get("rag_context", [])[:3],
            }
            for profile in profiles
            if isinstance(profile, dict)
        ]
        prompt = (
            "You are CompIntel's SWOT synthesizer.\n"
            "Return strict JSON with keys summary, competitors, cross_analysis.\n"
            "For each competitor, produce strengths, weaknesses, opportunities, threats arrays. "
            "Every item must be an object with text and evidence fields. Evidence must be derived "
            "from profile sources, URLs, scraped content, search results, or RAG context.\n"
            "cross_analysis must include common_strengths and differentiators.\n"
            f"Profiles: {safe_json_dumps(compact_profiles)}\n"
            f"Market analysis: {safe_json_dumps(market_analysis)}\n"
        )
        try:
            raw = await completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=2200,
                temperature=0.2,
            )
        except TypeError:
            raw = await completion_fn(prompt)
        except Exception:
            logger.exception("LLM call failed, returning None")
            return None

        parsed = load_repaired_json(str(raw))
        if isinstance(parsed, dict):
            return self._normalize_swot(parsed)
        return None

    def _normalize_swot(self, payload: dict[str, Any]) -> dict[str, Any]:
        competitors = []
        for competitor in payload.get("competitors", []):
            if not isinstance(competitor, dict):
                continue
            competitors.append(
                {
                    "name": competitor.get("name", "unknown"),
                    "strengths": self._normalize_items(competitor.get("strengths", [])),
                    "weaknesses": self._normalize_items(competitor.get("weaknesses", [])),
                    "opportunities": self._normalize_items(competitor.get("opportunities", [])),
                    "threats": self._normalize_items(competitor.get("threats", [])),
                }
            )
        cross = payload.get("cross_analysis") or {}
        return {
            "summary": str(payload.get("summary", "")),
            "competitors": competitors,
            "cross_analysis": {
                "common_strengths": self._normalize_items(cross.get("common_strengths", [])),
                "differentiators": self._normalize_items(cross.get("differentiators", [])),
            },
        }

    def _normalize_items(self, values: Any) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, dict):
                items.append(
                    {
                        "text": str(value.get("text", "")),
                        "evidence": str(value.get("evidence", "")),
                    }
                )
            else:
                items.append({"text": str(value), "evidence": ""})
        return [item for item in items if item["text"]]

    def _fallback_swot(self, profiles: list[dict[str, Any]], market_analysis: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": "placeholder SWOT analysis",
            "competitors": [
                {
                    "name": profile.get("name", "unknown"),
                    "strengths": [f"{profile.get('name', 'unknown')} strength"],
                    "weaknesses": [f"{profile.get('name', 'unknown')} weakness"],
                    "opportunities": [market_analysis.get("market_overview", "market opportunity")],
                    "threats": [market_analysis.get("barriers_to_entry", ["market threat"])[0]],
                }
                for profile in profiles
                if isinstance(profile, dict)
            ],
        }

    def _derived_swot(self, profiles: list[dict[str, Any]], market_analysis: dict[str, Any]) -> dict[str, Any]:
        overview = str(market_analysis.get("market_overview") or "Market context was derived from competitor profiles.")
        barriers = market_analysis.get("barriers_to_entry") or ["Enterprise switching costs"]
        competitors = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            name = str(profile.get("name", "unknown"))
            summary = str(profile.get("summary") or f"{name} has available profile evidence.")
            evidence = self._first_evidence(profile)
            competitors.append(
                {
                    "name": name,
                    "strengths": [{"text": summary, "evidence": evidence}],
                    "weaknesses": [
                        {
                            "text": "Public evidence is incomplete for pricing, traction, or enterprise adoption.",
                            "evidence": evidence,
                        }
                    ],
                    "opportunities": [{"text": overview, "evidence": evidence}],
                    "threats": [{"text": str(barriers[0] if isinstance(barriers, list) else barriers), "evidence": evidence}],
                }
            )
        return {
            "summary": "SWOT synthesized from available competitor profiles and market context.",
            "competitors": competitors,
            "cross_analysis": {
                "common_strengths": [
                    {
                        "text": "Competitors converge around flexible collaboration and workflow consolidation.",
                        "evidence": "profile summaries and market analysis",
                    }
                ],
                "differentiators": [
                    {
                        "text": "Differentiation depends on product breadth, integrations, ecosystem reach, and adoption friction.",
                        "evidence": "profile summaries and market analysis",
                    }
                ],
            },
        }

    def _first_evidence(self, profile: dict[str, Any]) -> str:
        snippets = (profile.get("search_results") or [])
        for result in (snippets if isinstance(snippets, list) else []):
            if isinstance(result, dict):
                text = (result.get("snippet") or "").strip()
                if text:
                    return text[:200]
        scraped = (profile.get("scraped_content") or [])
        for entry in (scraped if isinstance(scraped, list) else []):
            if isinstance(entry, dict):
                content = (entry.get("content") or "").strip()
                if content:
                    return content[:200]
        # fallback to source URLs / labels
        for key in ("search_results", "scraped_content", "rag_context", "sources"):
            values = profile.get(key, [])
            for value in values if isinstance(values, list) else [values]:
                if isinstance(value, dict):
                    source = value.get("url") or value.get("source") or value.get("title")
                    if source:
                        return str(source)
                elif str(value).strip() and str(value).strip() not in {"search_worker", "scrape_worker", "rag_retriever"}:
                    return str(value)
        return "profile summary"
