"""Report writing for CompIntel Research."""

from __future__ import annotations
import logging

from typing import Any

logger = logging.getLogger(__name__)

from ..llm import _split_provider_model
from ..parsing import load_repaired_json, safe_json_dumps
from ..settings import CompIntelSettings
from .base import BaseCompIntelAgent


class ReportWriterAgent(BaseCompIntelAgent):
    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model)
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        query = "unknown"
        intent = {}
        profiles = []
        market_analysis = {}
        swot_analysis = {}
        review_feedback = {}
        if isinstance(state, dict):
            query = state.get("query") or query
            intent = state.get("intent") or {}
            profiles = state.get("profiles") or []
            market_analysis = state.get("market_analysis") or {}
            swot_analysis = state.get("swot_analysis") or {}
            review_feedback = state.get("review_feedback") or {}

        settings = CompIntelSettings.from_env()
        report = await self._try_llm_write(
            query=query,
            intent=intent,
            profiles=profiles,
            market_analysis=market_analysis,
            swot_analysis=swot_analysis,
            review_feedback=review_feedback,
            settings=settings,
        )
        source = "llm"
        if report is None:
            if settings.llm_api_key:
                report = self._derived_report(query, intent, profiles, market_analysis, swot_analysis)
                source = "derived"
            else:
                report = self._fallback_report(query, intent, profiles, market_analysis, swot_analysis)
                source = "template"

        return {
            "report": report,
            "execution_log": [
                {"node": "report_writer", "event": "completed", "detail": f"{source}: report drafted"}
            ],
        }

    async def _try_llm_write(
        self,
        query: str,
        intent: dict[str, Any],
        profiles: list[dict[str, Any]],
        market_analysis: dict[str, Any],
        swot_analysis: dict[str, Any],
        review_feedback: dict[str, Any],
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
        sources = self._extract_sources(profiles)
        compact_profiles = [
            {
                "name": profile.get("name"),
                "website": profile.get("website"),
                "summary": profile.get("summary"),
                "sources": profile.get("sources", []),
                "search_results": profile.get("search_results", [])[:2],
                "rag_context": profile.get("rag_context", [])[:1],
            }
            for profile in profiles
            if isinstance(profile, dict)
        ]
        prompt = (
            "You are CompIntel's report writer. Write a concise Chinese competitive "
            "intelligence report from the provided evidence. Return strict JSON with "
            "keys: title, executive_summary, sections, conclusion, sources, data_gaps. "
            "executive_summary must be no more than 300 Chinese characters. Each "
            "section must be an object with title, content, key_insights. Every factual "
            "claim in content should include [Source: url] using the provided sources. "
            "Do not invent facts that are absent from the inputs; put missing items in "
            "data_gaps. Avoid the word placeholder.\n"
            f"Query: {query}\n"
            f"Intent: {safe_json_dumps(intent)}\n"
            f"Profiles: {safe_json_dumps(compact_profiles)}\n"
            f"Market analysis: {safe_json_dumps(market_analysis)}\n"
            f"SWOT analysis: {safe_json_dumps(swot_analysis)}\n"
            f"Available sources: {safe_json_dumps(sources)}\n"
            f"Prior review feedback, if any: {safe_json_dumps(review_feedback)}\n"
        )
        try:
            raw = await completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=2600,
                temperature=0.2,
            )
        except TypeError:
            raw = await completion_fn(prompt)
        except Exception:
            logger.exception("LLM call failed, returning None")
            return None

        parsed = load_repaired_json(str(raw))
        if isinstance(parsed, dict):
            return self._normalize_report(parsed, query, intent, profiles)
        return None

    def _normalize_report(
        self,
        payload: dict[str, Any],
        query: str,
        intent: dict[str, Any],
        profiles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sources = self._normalize_sources(payload.get("sources")) or self._extract_sources(profiles)
        sections = []
        for section in payload.get("sections", []):
            if isinstance(section, dict):
                content = str(section.get("content", ""))
                if content and "[Source:" not in content and sources:
                    content = f"{content} [Source: {sources[0]}]"
                sections.append(
                    {
                        "title": str(section.get("title", "Untitled")),
                        "content": content,
                        "key_insights": self._normalize_list(section.get("key_insights", [])),
                    }
                )
            elif str(section).strip():
                sections.append(
                    {
                        "title": str(section).strip(),
                        "content": "",
                        "key_insights": [],
                    }
                )

        return {
            "title": str(payload.get("title") or f"CompIntel Report: {query}"),
            "executive_summary": str(
                payload.get("executive_summary") or f"Analysis for {intent.get('target', 'unknown')}"
            )[:300],
            "sections": sections,
            "conclusion": str(payload.get("conclusion", "")),
            "sources": sources,
            "data_gaps": self._normalize_list(payload.get("data_gaps", [])),
        }

    def _fallback_report(
        self,
        query: str,
        intent: dict[str, Any],
        profiles: list[dict[str, Any]],
        market_analysis: dict[str, Any],
        swot_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        sources = self._extract_sources(profiles)
        first_source = sources[0] if sources else ""
        profile_lines = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            summary = str(profile.get("summary") or "")
            citation = f" [Source: {first_source}]" if first_source and summary else ""
            profile_lines.append(f"{profile.get('name', 'unknown')}: {summary}{citation}")

        market_content = self._summarize_market(market_analysis)
        swot_content = self._summarize_swot(swot_analysis)
        if first_source:
            if market_content and "[Source:" not in market_content:
                market_content = f"{market_content} [Source: {first_source}]"
            if swot_content and "[Source:" not in swot_content:
                swot_content = f"{swot_content} [Source: {first_source}]"

        data_gaps = []
        if not sources:
            data_gaps.append("No source URLs were available in competitor profiles.")
        if not profiles:
            data_gaps.append("No competitor profiles were available.")
        if not market_analysis:
            data_gaps.append("No market analysis was available.")
        if not swot_analysis:
            data_gaps.append("No SWOT analysis was available.")

        sections = [
            {
                "title": "Competitor Profiles",
                "content": "\n".join(profile_lines),
                "key_insights": [profile.get("summary", "") for profile in profiles if isinstance(profile, dict)],
            },
            {
                "title": "Market Analysis",
                "content": market_content,
                "key_insights": self._normalize_list(market_analysis.get("growth_trends", [])),
            },
            {
                "title": "SWOT",
                "content": swot_content,
                "key_insights": self._normalize_list([swot_analysis.get("summary", "")]),
            },
        ]
        target = intent.get("target", "unknown")
        return {
            "title": f"CompIntel Report: {query}",
            "executive_summary": f"Analysis for {target}"[:300],
            "profiles": profiles,
            "market_analysis": market_analysis,
            "swot_analysis": swot_analysis,
            "sections": sections,
            "conclusion": "",
            "sources": sources,
            "data_gaps": data_gaps,
        }

    def _derived_report(
        self,
        query: str,
        intent: dict[str, Any],
        profiles: list[dict[str, Any]],
        market_analysis: dict[str, Any],
        swot_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        report = self._fallback_report(query, intent, profiles, market_analysis, swot_analysis)
        target = intent.get("target", "unknown")
        sources = report.get("sources", [])
        citation = f" [Source: {sources[0]}]" if sources else ""
        report["executive_summary"] = (
            f"{target} competitive analysis based on collected profiles, market context, and SWOT evidence."
        )[:300]
        report["conclusion"] = self._build_derived_conclusion(target, profiles, sources, citation)
        report["data_gaps"] = [
            "Validate the latest revenue, customer count, and pricing details with authoritative sources.",
            *report.get("data_gaps", []),
        ]
        return report

    @staticmethod
    def _build_derived_conclusion(target: str, profiles: list[dict[str, Any]],
                                   sources: list[str], citation: str) -> str:
        total_search = sum(len(p.get("search_results", [])) for p in profiles if isinstance(p, dict))
        total_scraped = sum(len(p.get("scraped_content", [])) for p in profiles if isinstance(p, dict))
        names = [p.get("name", "") for p in profiles if isinstance(p, dict) and p.get("name")]
        competitor_str = ", ".join(names[:3])
        if len(names) > 3:
            competitor_str += f" and {len(names) - 3} more"
        return (
            f"Analysis of {target} based on {total_search} search results and "
            f"{total_scraped} scraped pages across {len(profiles)} competitors"
            f"{' (' + competitor_str + ')' if competitor_str else ''}. "
            f"Further LLM-powered analysis is recommended for deeper strategic insights."
            f"{citation}"
        )[:400]

    def _extract_sources(self, profiles: list[dict[str, Any]]) -> list[str]:
        sources: list[str] = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            for item_key in ("search_results", "scraped_content", "rag_context"):
                for value in profile.get(item_key, []):
                    self._append_source(sources, value)
            for value in profile.get("sources", []):
                self._append_source(sources, value)
        return sources

    def _append_source(self, sources: list[str], value: Any) -> None:
        source = ""
        if isinstance(value, dict):
            source = str(value.get("url") or value.get("source") or value.get("title") or "")
        else:
            source = str(value)
        source = source.strip()
        if source in {"search_worker", "scrape_worker", "rag_retriever"}:
            return
        if source and source not in sources:
            sources.append(source)

    def _normalize_sources(self, values: Any) -> list[str]:
        sources: list[str] = []
        for value in values if isinstance(values, list) else [values]:
            self._append_source(sources, value)
        return sources

    def _normalize_list(self, values: Any) -> list[str]:
        if isinstance(values, list):
            return [str(value) for value in values if str(value).strip()]
        if values:
            return [str(values)]
        return []

    def _summarize_market(self, market_analysis: dict[str, Any]) -> str:
        overview = str(market_analysis.get("market_overview", "")).strip()
        trends = self._normalize_list(market_analysis.get("growth_trends", []))
        differentiators = self._normalize_list(market_analysis.get("key_differentiators", []))
        parts = []
        if overview:
            parts.append(overview)
        if trends:
            parts.append("Growth trends: " + "; ".join(trends[:3]))
        if differentiators:
            parts.append("Key differentiators: " + "; ".join(differentiators[:3]))
        return "\n".join(parts) or "Market analysis is not available."

    def _summarize_swot(self, swot_analysis: dict[str, Any]) -> str:
        summary = str(swot_analysis.get("summary", "")).strip()
        competitors = [
            str(item.get("name", "unknown"))
            for item in swot_analysis.get("competitors", [])
            if isinstance(item, dict)
        ]
        parts = []
        if summary:
            parts.append(summary)
        if competitors:
            parts.append("Competitors covered: " + ", ".join(competitors))
        return "\n".join(parts) or "SWOT analysis is not available."
