"""Market analysis for CompIntel Research."""

from __future__ import annotations
import logging

from typing import Any

logger = logging.getLogger(__name__)

from ..llm import _split_provider_model
from ..parsing import load_repaired_json, safe_json_dumps
from ..settings import CompIntelSettings
from .base import BaseCompIntelAgent


class MarketAnalystAgent(BaseCompIntelAgent):
    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model)
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        profiles = []
        market_segment = "unknown"
        if isinstance(state, dict):
            profiles = state.get("profiles") or state.get("competitor_profiles") or []
            market_segment = state.get("market_segment") or market_segment

        settings = CompIntelSettings.from_env()
        market_analysis = await self._try_llm_analyze(profiles, market_segment, settings)
        source = "llm"
        if market_analysis is None:
            if settings.llm_api_key:
                market_analysis = self._derived_analysis(profiles, market_segment)
                source = "derived"
            else:
                market_analysis = self._fallback_analysis(profiles, market_segment)
                source = "template"

        return {
            "market_analysis": market_analysis,
            "execution_log": [
                {"node": "market_analyst", "event": "completed", "detail": f"{source}: {market_segment}"}
            ],
        }

    async def _try_llm_analyze(
        self,
        profiles: list[dict[str, Any]],
        market_segment: str,
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

        provider, model = _split_provider_model(settings.smart_llm)
        compact_profiles = [
            {
                "name": profile.get("name"),
                "summary": profile.get("summary"),
                "sources": profile.get("sources", []),
                "search_results": profile.get("search_results", [])[:2],
                "rag_context": profile.get("rag_context", [])[:1],
            }
            for profile in profiles
            if isinstance(profile, dict)
        ]
        prompt = (
            "You are CompIntel's market analyst.\n"
            "Analyze the market from competitor profiles and return strict JSON with keys: "
            "market_overview, growth_trends, competitive_landscape, key_differentiators, "
            "barriers_to_entry.\n"
            "competitive_landscape must include leaders, challengers, and niche lists.\n"
            f"Market segment: {market_segment}\n"
            f"Profiles: {safe_json_dumps(compact_profiles)}\n"
        )
        try:
            raw = await completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=1800,
                temperature=0.2,
            )
        except TypeError:
            raw = await completion_fn(prompt)
        except Exception:
            logger.exception("LLM call failed, returning None")
            return None

        parsed = load_repaired_json(str(raw))
        if isinstance(parsed, dict):
            return self._normalize_analysis(parsed)
        return None

    @staticmethod
    def _normalize_list_field(value: Any) -> list[str]:
        """Return a list of strings regardless of whether the LLM returned a string or a list."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    @staticmethod
    def _normalize_list_of_strings(value: Any) -> list[str]:
        """Handle LLM returning either list[str] or a single string for list fields."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("company") or "")
                    if name.strip():
                        result.append(name.strip())
                elif str(item).strip():
                    result.append(str(item).strip())
            return result
        return []

    def _normalize_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        landscape = payload.get("competitive_landscape") or {}
        return {
            "market_overview": str(payload.get("market_overview", "")),
            "growth_trends": self._normalize_list_field(payload.get("growth_trends", [])),
            "competitive_landscape": {
                "leaders": self._normalize_list_of_strings(landscape.get("leaders", landscape.get("market_leaders", []))),
                "challengers": self._normalize_list_of_strings(landscape.get("challengers", [])),
                "niche": self._normalize_list_of_strings(landscape.get("niche", landscape.get("niche_players", []))),
            },
            "key_differentiators": self._normalize_list_field(payload.get("key_differentiators", [])),
            "barriers_to_entry": self._normalize_list_field(payload.get("barriers_to_entry", [])),
        }

    def _fallback_analysis(self, profiles: list[dict[str, Any]], market_segment: str) -> dict[str, Any]:
        return {
            "market_overview": f"Market overview for {market_segment}.",
            "market_size_estimate": "unknown",
            "growth_trends": ["placeholder growth trend"],
            "competitive_landscape": {
                "market_leaders": [profile.get("name") for profile in profiles[:2] if isinstance(profile, dict)],
                "challengers": [profile.get("name") for profile in profiles[2:4] if isinstance(profile, dict)],
                "niche_players": [profile.get("name") for profile in profiles[4:] if isinstance(profile, dict)],
            },
            "key_differentiators": ["placeholder differentiator"],
            "technology_trends": ["placeholder technology trend"],
            "barriers_to_entry": ["placeholder barrier"],
        }

    def _derived_analysis(self, profiles: list[dict[str, Any]], market_segment: str) -> dict[str, Any]:
        names = [str(profile.get("name", "unknown")) for profile in profiles if isinstance(profile, dict)]
        summaries = [
            str(profile.get("summary", "")).strip()
            for profile in profiles
            if isinstance(profile, dict) and str(profile.get("summary", "")).strip()
        ]
        segment = market_segment or "target market"
        trends = self._extract_trends_from_snippets(profiles)
        return {
            "market_overview": (
                f"{segment} includes {', '.join(names) if names else 'the profiled competitors'}; "
                f"available evidence emphasizes product positioning, distribution, and collaboration workflows."
            ),
            "growth_trends": trends or [
                "AI-assisted workflows and knowledge management",
                "Integrated collaboration across documents, databases, and team communication",
            ],
            "competitive_landscape": {
                "leaders": names[:2],
                "challengers": names[2:4],
                "niche": names[4:],
            },
            "key_differentiators": summaries[:3] or ["Product scope, ecosystem integrations, and workflow depth"],
            "barriers_to_entry": [
                "Enterprise switching costs",
                "Trust, security, and integration requirements",
            ],
        }

    @staticmethod
    def _extract_trends_from_snippets(profiles: list[dict[str, Any]]) -> list[str]:
        _STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
                        "at", "to", "for", "with", "by", "as", "it", "its", "and", "or",
                        "that", "this", "has", "have", "from", "but", "not", "be", "will"}
        snippets = []
        for profile in profiles:
            for result in (profile.get("search_results") or [])[:5]:
                text = (result.get("snippet") or "").strip()
                if text:
                    snippets.append(text)
        bigrams: dict[str, int] = {}
        for text in snippets:
            words = text.lower().split()
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                if w1 in _STOP_WORDS or w2 in _STOP_WORDS:
                    continue
                if len(w1) < 3 or len(w2) < 3:
                    continue
                bigram = f"{w1} {w2}"
                # uppercase-priority: bump score if original words were capitalised
                orig1 = text.split()[i] if i < len(text.split()) else w1
                orig2 = text.split()[i + 1] if i + 1 < len(text.split()) else w2
                boost = 2 if (orig1[0].isupper() and orig2[0].isupper()) else 1
                bigrams[bigram] = bigrams.get(bigram, 0) + boost
        return [bg for bg, cnt in sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
                if cnt >= 2][:5]
