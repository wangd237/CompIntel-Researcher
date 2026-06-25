"""Market analysis for CompIntel Research."""

from __future__ import annotations
import logging

from typing import Any

logger = logging.getLogger(__name__)

from ..parsing import safe_json_dumps
from ..prompts import load_prompt
from .base import BaseCompIntelAgent


class MarketAnalystAgent(BaseCompIntelAgent):
    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model, model_key="smart")
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        profiles = s.profiles
        market_segment = s.market_segment

        market_analysis = await self._try_llm_analyze(profiles, market_segment)
        source = "llm"
        if market_analysis is None:
            settings = self.llm.settings
            if settings.llm_api_key:
                market_analysis = self._derived_analysis(profiles, market_segment)
                source = "derived"
            else:
                market_analysis = self._fallback_analysis(profiles, market_segment)
                source = "template"

        # ── self-audit (LLM path only; non-blocking) ──
        audit_warnings: list[str] = []
        if source == "llm":
            audit_warnings = self._self_audit(market_analysis, profiles)

        return {
            "market_analysis": market_analysis,
            "execution_log": [
                {"node": "market_analyst", "event": "completed",
                 "detail": f"{source}: {market_segment}"}
            ],
            "warnings": audit_warnings,
        }

    async def _try_llm_analyze(
        self,
        profiles: list[dict[str, Any]],
        market_segment: str,
    ) -> dict[str, Any] | None:
        if self.completion_fn is not None:
            return await self._legacy_llm_analyze(profiles, market_segment)

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
        prompt = load_prompt("market_analyst")
        parsed = await self.llm.call_and_parse(
            prompt.format(
                market_segment=market_segment,
                profiles=safe_json_dumps(compact_profiles),
            ),
            model_key=prompt.model_key,
            max_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
        )
        if isinstance(parsed, dict):
            return self._normalize_analysis(parsed)
        return None

    async def _legacy_llm_analyze(
        self,
        profiles: list[dict[str, Any]],
        market_segment: str,
    ) -> dict[str, Any] | None:
        """Backward-compat path when a test-injected *completion_fn* is present."""
        try:
            from ..llm import create_chat_completion, _split_provider_model
            from ..settings import CompIntelSettings
        except Exception:
            logger.exception("Failed to import legacy LLM deps")
            return None

        settings = CompIntelSettings.from_env()
        if not settings.llm_api_key:
            return None

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
            "Analyze the SPECIFIC market segment named below and return strict JSON "
            "with keys: market_overview, growth_trends, competitive_landscape, "
            "key_differentiators, barriers_to_entry.\n"
            "competitive_landscape must include leaders, challengers, and niche lists.\n"
            "CRITICAL: Every trend, differentiator, and barrier MUST be specific to "
            "this exact market segment — do NOT use generic SaaS/collaboration language "
            "unless the segment IS SaaS/collaboration.\n"
            f"Market segment: {market_segment}\n"
            f"Profiles: {safe_json_dumps(compact_profiles)}\n"
        )
        try:
            raw = await self.completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=1800,
                temperature=0.2,
            )
        except TypeError:
            raw = await self.completion_fn(prompt)
        except Exception as exc:
            logger.warning("Market analyst LLM call failed; using derived analysis: %s", exc)
            return None

        from ..parsing import load_repaired_json
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

    @staticmethod
    def _clean_artifact(val: str) -> str:
        """P2-3: Strip JSON artifact suffixes (``}}``, ``}``, ``]``) that LLM
        output sometimes trails at the end of string values when the response is
        truncated or repaired."""
        cleaned = val.strip()
        while cleaned.endswith(("}}", "} ]", "}]", "}", "]")):
            if cleaned.endswith("}}"):
                cleaned = cleaned[:-2].rstrip()
            elif cleaned.endswith("}]"):
                cleaned = cleaned[:-2].rstrip()
            elif cleaned.endswith("} ]"):
                cleaned = cleaned[:-3].rstrip()
            else:
                cleaned = cleaned[:-1].rstrip()
        return cleaned

    def _normalize_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        landscape = payload.get("competitive_landscape") or {}
        return {
            "market_overview": self._clean_artifact(str(payload.get("market_overview", ""))),
            "growth_trends": [self._clean_artifact(s) for s in self._normalize_list_field(payload.get("growth_trends", []))],
            "competitive_landscape": {
                "leaders": [self._clean_artifact(s) for s in self._normalize_list_of_strings(landscape.get("leaders", landscape.get("market_leaders", [])))],
                "challengers": [self._clean_artifact(s) for s in self._normalize_list_of_strings(landscape.get("challengers", []))],
                "niche": [self._clean_artifact(s) for s in self._normalize_list_of_strings(landscape.get("niche", landscape.get("niche_players", [])))],
            },
            "key_differentiators": [self._clean_artifact(s) for s in self._normalize_list_field(payload.get("key_differentiators", []))],
            "barriers_to_entry": [self._clean_artifact(s) for s in self._normalize_list_field(payload.get("barriers_to_entry", []))],
        }

    # ── self-audit ─────────────────────────────────────────────────────

    @staticmethod
    def _self_audit(
        analysis: dict[str, Any], profiles: list[dict[str, Any]]
    ) -> list[str]:
        """Validate LLM-generated market analysis against input profiles.

        Pure set operations — no LLM call, < 1 ms.  Returns a list of
        warning strings (empty when clean).
        """
        warnings: list[str] = []
        profile_names = {
            str(p.get("name", "")).strip()
            for p in profiles if isinstance(p, dict) and str(p.get("name", "")).strip()
        }
        if not profile_names:
            return warnings

        landscape = analysis.get("competitive_landscape", {})
        leaders = set(landscape.get("leaders", []))
        challengers = set(landscape.get("challengers", []))
        niche = set(landscape.get("niche", []))

        all_mentioned = leaders | challengers | niche
        if not all_mentioned:
            return warnings

        # 1. Phantom companies: names the LLM invented that are NOT in profiles
        unknown = all_mentioned - profile_names
        if unknown:
            warnings.append(
                f"Market analysis references {len(unknown)} company name(s) not in "
                f"input profiles: {sorted(unknown)}. These may be LLM hallucinations."
            )

        # 2. Duplicate classification: same company in >1 bucket
        dupes = (leaders & challengers) | (leaders & niche) | (challengers & niche)
        if dupes:
            warnings.append(
                f"Companies appear in multiple landscape categories: "
                f"{sorted(dupes)}. Each company should be in exactly one bucket."
            )

        return warnings

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
            and not str(profile.get("summary", "")).startswith("Profile summary for")
        ]
        segment = market_segment or "target market"
        trends = self._extract_trends_from_snippets(profiles)
        overview_parts = [f"{segment} includes {', '.join(names) if names else 'the profiled competitors'}."]
        if summaries:
            overview_parts.append(f"Key themes from profiles: {'; '.join(summaries[:3])}.")
        else:
            overview_parts.append(
                "Profile data was insufficient for a detailed market overview. "
                "Consider re-running with LLM enabled or broadening the search scope."
            )
        return {
            "market_overview": " ".join(overview_parts),
            "growth_trends": trends or [
                "Insufficient search data to extract market trends — consider enabling LLM analysis.",
            ],
            "competitive_landscape": {
                "leaders": names[:2],
                "challengers": names[2:4],
                "niche": names[4:],
            },
            "key_differentiators": summaries[:3] or [
                "Insufficient profile data to determine key differentiators.",
            ],
            "barriers_to_entry": [
                "Insufficient data to determine barriers to entry.",
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
