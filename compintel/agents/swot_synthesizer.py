"""SWOT synthesis for CompIntel Research."""

from __future__ import annotations

from typing import Any

from .base import BaseCompIntelAgent


class SWOTSynthesizerAgent(BaseCompIntelAgent):
    async def __call__(self, state: Any) -> dict[str, Any]:
        profiles = []
        market_analysis = {}
        if isinstance(state, dict):
            profiles = state.get("profiles") or []
            market_analysis = state.get("market_analysis") or {}

        swot = {
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
        return {
            "swot_analysis": swot,
            "execution_log": [
                {"node": "swot_synthesizer", "event": "completed", "detail": "swot synthesized"}
            ],
        }
