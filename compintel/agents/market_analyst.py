"""Market analysis for CompIntel Research."""

from __future__ import annotations

from typing import Any

from .base import BaseCompIntelAgent


class MarketAnalystAgent(BaseCompIntelAgent):
    async def __call__(self, state: Any) -> dict[str, Any]:
        profiles = []
        market_segment = "unknown"
        if isinstance(state, dict):
            profiles = state.get("profiles") or state.get("competitor_profiles") or []
            market_segment = state.get("market_segment") or market_segment

        market_analysis = {
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
        return {
            "market_analysis": market_analysis,
            "execution_log": [
                {"node": "market_analyst", "event": "completed", "detail": market_segment}
            ],
        }
