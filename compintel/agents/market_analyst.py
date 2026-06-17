"""Market analysis for CompIntel Research."""

from __future__ import annotations

from typing import Any

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
                from gpt_researcher.utils.llm import create_chat_completion
            except Exception:
                return None
            completion_fn = create_chat_completion

        provider, model = self._split_provider_model(settings.smart_llm)
        compact_profiles = [
            {
                "name": profile.get("name"),
                "summary": profile.get("summary"),
                "sources": profile.get("sources", []),
                "search_results": profile.get("search_results", [])[:5],
                "rag_context": profile.get("rag_context", [])[:3],
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
            return None

        parsed = load_repaired_json(str(raw))
        if isinstance(parsed, dict):
            return self._normalize_analysis(parsed)
        return None

    def _normalize_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        landscape = payload.get("competitive_landscape") or {}
        return {
            "market_overview": str(payload.get("market_overview", "")),
            "growth_trends": [str(item) for item in payload.get("growth_trends", [])],
            "competitive_landscape": {
                "leaders": list(landscape.get("leaders", landscape.get("market_leaders", []))),
                "challengers": list(landscape.get("challengers", [])),
                "niche": list(landscape.get("niche", landscape.get("niche_players", []))),
            },
            "key_differentiators": [str(item) for item in payload.get("key_differentiators", [])],
            "barriers_to_entry": [str(item) for item in payload.get("barriers_to_entry", [])],
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

    def _split_provider_model(self, value: str) -> tuple[str, str]:
        if ":" in value:
            provider, model = value.split(":", 1)
            return provider.strip() or "openai", model.strip() or "gpt-4o-mini"
        return "openai", value.strip() or "gpt-4o-mini"
