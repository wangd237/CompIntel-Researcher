"""SWOT synthesis for CompIntel Research.

Per-competitor analysis: instead of one monolithic call for all N competitors
(which squeezes 20 SWOT cells into 2200 tokens), each competitor gets an
independent LLM call with its FULL curated profile.  After per-competitor
SWOTs are collected, a lightweight aggregation call produces cross_analysis
and summary."""

from __future__ import annotations
import asyncio
import logging

from typing import Any

logger = logging.getLogger(__name__)

from ..parsing import safe_json_dumps
from ..prompts import load_prompt
from .base import BaseCompIntelAgent

# Instructions injected into per-competitor SWOT prompts based on curator's
# evidence grade.  The goal is to stop the LLM from fabricating claims when
# data for W/T/O is genuinely unavailable.
_GRADE_INSTRUCTIONS: dict[str, str] = {
    "rich": (
        "DATA QUALITY: RICH — substantial search, scrape, and RAG data available. "
        "Provide 2-3 well-evidenced items per SWOT quadrant.\n"
    ),
    "adequate": (
        "DATA QUALITY: ADEQUATE — moderate data. Provide 2 items for S/O with evidence. "
        "W/T may have limited evidence — note gaps honestly, do not fabricate.\n"
    ),
    "thin": (
        "DATA QUALITY: THIN — sparse data. Output 1-2 items for S only if evidence exists. "
        "For W/T/O: if no specific evidence, output a single item stating "
        "'insufficient data for this dimension'.\n"
    ),
    "empty": (
        "DATA QUALITY: EMPTY — no real data collected. Output ONE item per quadrant "
        "stating 'no data available'. Do NOT invent any claims.\n"
    ),
}


class SWOTSynthesizerAgent(BaseCompIntelAgent):
    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model, model_key="strategic")
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        profiles = s.profiles
        market_analysis = s.market_analysis

        swot = await self._try_llm_synthesize(profiles, market_analysis)
        source = "llm"
        if swot is None:
            settings = self.llm.settings
            if settings.llm_api_key:
                swot = self._derived_swot(profiles, market_analysis)
                source = "derived"
            else:
                swot = self._fallback_swot(profiles, market_analysis)
                source = "template"

        # ── self-audit (LLM path only; non-blocking) ──
        audit_warnings: list[str] = []
        if source == "llm":
            audit_warnings = self._self_audit(swot, profiles)

        return {
            "swot_analysis": swot,
            "execution_log": [
                {"node": "swot_synthesizer", "event": "completed",
                 "detail": f"{source}: swot synthesized"}
            ],
            "warnings": audit_warnings,
        }

    async def _try_llm_synthesize(
        self,
        profiles: list[dict[str, Any]],
        market_analysis: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.completion_fn is not None:
            return await self._legacy_llm_synthesize(profiles, market_analysis)

        # ── Per-competitor SWOT ────────────────────────────────────────
        sem = asyncio.Semaphore(3)  # throttle concurrent per-competitor calls
        market_overview = str(market_analysis.get("market_overview", ""))

        async def _swot_for(profile: dict[str, Any]) -> dict[str, Any] | None:
            async with sem:
                return await self._synthesize_one(profile, market_overview)

        tasks = [
            _swot_for(profile)
            for profile in profiles
            if isinstance(profile, dict)
        ]
        if not tasks:
            return None

        results = await asyncio.gather(*tasks)

        competitors = []
        for result in results:
            if isinstance(result, dict):
                competitors.append(result)

        if not competitors:
            return None

        # ── Cross-analysis (aggregation) ───────────────────────────────
        cross = await self._synthesize_cross_analysis(competitors, market_overview)

        return {
            "summary": cross.get("summary", f"SWOT synthesized for {len(competitors)} competitors."),
            "competitors": competitors,
            "cross_analysis": cross.get("cross_analysis", {}),
        }

    async def _synthesize_one(
        self, profile: dict[str, Any], market_overview: str
    ) -> dict[str, Any] | None:
        """Generate SWOT for a single competitor with its full curated profile."""
        name = profile.get("name", "unknown")
        evidence_grade = profile.get("evidence_grade", "adequate")
        curator_note = profile.get("curator_note", "")

        # Pass the FULL profile — we have tokens now since it's just one competitor
        compact = {
            "name": name,
            "summary": profile.get("summary"),
            "sources": profile.get("sources", [])[:6],
            "search_results": [
                {"title": r.get("title"), "snippet": r.get("snippet") or r.get("body")}
                for r in (profile.get("search_results") or [])[:6]
                if isinstance(r, dict)
            ],
            "scraped_content": [
                {"url": r.get("url"), "title": r.get("title"), "content": str(r.get("content") or r.get("raw_content", ""))[:800]}
                for r in (profile.get("scraped_content") or [])[:3]
                if isinstance(r, dict)
            ],
            "rag_context": [
                {"text": str(r.get("text", ""))[:600]}
                for r in (profile.get("rag_context") or [])[:3]
                if isinstance(r, dict)
            ],
        }

        grade_instruction = _GRADE_INSTRUCTIONS.get(evidence_grade, "")
        curator_line = f"\nCurator note: {curator_note}" if curator_note else ""

        prompt = load_prompt("swot_competitor")
        parsed = await self.llm.call_with_reasoning(
            prompt.format(
                name=name,
                market_overview=market_overview,
                grade_instruction=grade_instruction,
                curator_line=curator_line,
                profile_data=safe_json_dumps(compact),
            ),
            reasoning_max_tokens=2000, formatting_max_tokens=1200, temperature=prompt.temperature,
        )
        if isinstance(parsed, dict):
            return self._normalize_competitor_swot(parsed)
        return None

    async def _synthesize_cross_analysis(
        self, competitors: list[dict[str, Any]], market_overview: str
    ) -> dict[str, Any]:
        """Aggregate per-competitor SWOTs into cross_analysis and overall summary."""
        summaries = [
            {
                "name": c.get("name"),
                "strengths": [s.get("text") for s in (c.get("strengths") or [])],
                "weaknesses": [w.get("text") for w in (c.get("weaknesses") or [])],
            }
            for c in competitors
        ]
        prompt = load_prompt("swot_cross")
        parsed = await self.llm.call_with_reasoning(
            prompt.format(
                market_overview=market_overview,
                competitor_summaries=safe_json_dumps(summaries),
            ),
            reasoning_max_tokens=1500, formatting_max_tokens=800, temperature=prompt.temperature,
        )
        if isinstance(parsed, dict):
            return parsed
        return {
            "summary": f"SWOT analysis for {len(competitors)} competitors.",
            "cross_analysis": {"common_strengths": [], "differentiators": []},
        }

    async def _legacy_llm_synthesize(
        self,
        profiles: list[dict[str, Any]],
        market_analysis: dict[str, Any],
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

        provider, model = _split_provider_model(settings.strategic_llm)
        compact_profiles = [
            {
                "name": profile.get("name"),
                "summary": profile.get("summary"),
                "sources": profile.get("sources", []),
                "search_results": profile.get("search_results", [])[:2],
                "scraped_content": profile.get("scraped_content", [])[:1],
                "rag_context": profile.get("rag_context", [])[:1],
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
            raw = await self.completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=2200,
                temperature=0.2,
            )
        except TypeError:
            raw = await self.completion_fn(prompt)
        except Exception as exc:
            logger.warning("SWOT synthesizer LLM call failed; using derived SWOT: %s", exc)
            return None

        from ..parsing import load_repaired_json
        parsed = load_repaired_json(str(raw))
        if isinstance(parsed, dict):
            return self._normalize_swot(parsed)
        return None

    # ── self-audit ─────────────────────────────────────────────────────

    @staticmethod
    def _self_audit(
        swot: dict[str, Any], profiles: list[dict[str, Any]]
    ) -> list[str]:
        """Validate LLM-generated SWOT output against input profiles.

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

        swot_competitors = swot.get("competitors", [])
        swot_names = {
            str(c.get("name", "")).strip()
            for c in swot_competitors
            if isinstance(c, dict) and str(c.get("name", "")).strip()
        }
        if not swot_names:
            warnings.append("SWOT output contains no competitor entries.")
            return warnings

        # 1. Phantom competitors
        unknown = swot_names - profile_names
        if unknown:
            warnings.append(
                f"SWOT includes {len(unknown)} competitor(s) not in input profiles: "
                f"{sorted(unknown)}. These may be LLM hallucinations."
            )

        # 2. Missing competitors (LLM dropped some from input)
        missing = profile_names - swot_names
        if missing:
            warnings.append(
                f"SWOT is missing {len(missing)} competitor(s) from input profiles: "
                f"{sorted(missing)}. The pipeline delivered them but the LLM omitted them."
            )

        return warnings

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

    def _normalize_competitor_swot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize a single-competitor SWOT response."""
        return {
            "name": str(payload.get("name", "unknown")),
            "strengths": self._normalize_items(payload.get("strengths", [])),
            "weaknesses": self._normalize_items(payload.get("weaknesses", [])),
            "opportunities": self._normalize_items(payload.get("opportunities", [])),
            "threats": self._normalize_items(payload.get("threats", [])),
        }

    def _fallback_swot(self, profiles: list[dict[str, Any]], market_analysis: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": "placeholder SWOT analysis",
            "competitors": [
                {
                    "name": profile.get("name", "unknown"),
                    "strengths": [f"{profile.get('name', 'unknown')} strength"],
                    "weaknesses": [f"{profile.get('name', 'unknown')} weakness"],
                    "opportunities": [market_analysis.get("market_overview", "market opportunity")],
                    "threats": [(market_analysis.get("barriers_to_entry") or ["market threat"])[0]],
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
