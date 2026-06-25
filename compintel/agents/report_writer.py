"""Report writing for CompIntel Research.

Section-by-section assembly: instead of one monolithic LLM call that outputs a
deeply nested JSON, the report is built from 4 independent small calls + 1
pure-code assembler.  Each call handles one section; failure is isolated to
that section and never cascades to the whole report.
"""

from __future__ import annotations

import asyncio
import logging

from typing import Any

logger = logging.getLogger(__name__)

from ..llm import _split_provider_model
from ..parsing import load_repaired_json, safe_json_dumps
from ..settings import CompIntelSettings
from .base import BaseCompIntelAgent

_LANG_INSTRUCTION = {
    "zh": (
        "你必须用中文撰写这份竞品情报报告。"
        "executive_summary 不超过 300 个中文字符。"
        "每条事实声明必须包含 [Source: url] 标注。"
    ),
    "en": (
        "Write this report in English. "
        "Every factual claim must include [Source: url] using the provided sources. "
        "Do not invent facts absent from the inputs; put missing items in data_gaps."
    ),
}

_FALLBACK_EXECUTIVE_SUMMARY = {
    "zh": "本报告基于已收集的竞品画像、市场背景和 SWOT 证据进行的综合分析。",
    "en": "{target} competitive analysis based on collected profiles, market context, and SWOT evidence.",
}

_FALLBACK_CONCLUSION = {
    "zh": (
        "现有证据仅勾画了竞争轮廓。建议获取{target}及主要竞争对手的最新财务数据、"
        "市场份额和产品路线图，以支持更精确的战略决策。"
    ),
    "en": (
        "{target} should be evaluated against competitor product scope, "
        "workflow depth, ecosystem reach, and switching-cost barriers."
    ),
}


class ReportWriterAgent(BaseCompIntelAgent):
    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model, model_key="smart")
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        query = s.query
        intent = s.intent
        profiles = s.profiles
        market_analysis = s.market_analysis
        swot_analysis = s.swot_analysis
        review_feedback = s.review_feedback
        language = s.language

        settings = self.llm.settings
        report = await self._try_llm_write(
            query=query,
            intent=intent,
            profiles=profiles,
            market_analysis=market_analysis,
            swot_analysis=swot_analysis,
            review_feedback=review_feedback,
            language=language,
            settings=settings,
        )
        source = "llm"
        if report is None:
            if settings.llm_api_key:
                report = self._derived_report(query, intent, profiles, market_analysis, swot_analysis, language)
                source = "derived"
            else:
                report = self._fallback_report(query, intent, profiles, market_analysis, swot_analysis, language)
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
        language: str,
        settings: CompIntelSettings,
    ) -> dict[str, Any] | None:
        """Section-by-section assembly: 4 small LLM calls + 1 pure-code assembler.

        Each section is generated independently with a small prompt and small
        output window.  A single section failure only degrades that section —
        the rest of the report retains LLM quality.
        """
        if not settings.llm_api_key and self.completion_fn is None:
            return None

        completion_fn = self.completion_fn
        provider, model = None, None
        if completion_fn is None:
            # Resolve via LLMService — build an adapter that matches
            # create_chat_completion's signature for section generators.
            from ..llm import _split_provider_model
            provider, model = _split_provider_model(settings.smart_llm)
            async def _llm_call(**kwargs: Any) -> str:
                prompt = str(kwargs.get("messages", [{}])[0].get("content", ""))
                return await self.llm.call(
                    prompt,
                    model_key="smart",
                    max_tokens=int(kwargs.get("max_tokens", 500)),
                    temperature=float(kwargs.get("temperature", 0.3)),
                )
            completion_fn = _llm_call
        sources = self._extract_sources(profiles)
        lang_instruction = _LANG_INSTRUCTION.get(language, _LANG_INSTRUCTION["en"])
        target = intent.get("target", query)

        # Track per-section LLM success; return None only if *every* section
        # degraded, so the caller can switch to the full derived/fallback path.
        llm_success = False

        # ── 1. Executive summary (small call) ──
        exec_summary, exec_ok = await self._generate_executive_summary(
            target=target,
            market_analysis=market_analysis,
            swot_analysis=swot_analysis,
            profiles=profiles,
            review_feedback=review_feedback,
            sources=sources,
            lang_instruction=lang_instruction,
            language=language,
            provider=provider,
            model=model,
            completion_fn=completion_fn,
        )
        llm_success = llm_success or exec_ok

        # ── 2. Competitor narratives (parallel, one per competitor) ──
        narratives = await self._generate_competitor_narratives(
            profiles=profiles,
            sources=sources,
            lang_instruction=lang_instruction,
            language=language,
            provider=provider,
            model=model,
            completion_fn=completion_fn,
        )
        llm_success = llm_success or any(
            n.get("_llm") for n in narratives
        )

        # ── 3. Conclusion (small call) ──
        conclusion, conc_ok = await self._generate_conclusion(
            target=target,
            query=query,
            narratives=narratives,
            swot_analysis=swot_analysis,
            market_analysis=market_analysis,
            sources=sources,
            lang_instruction=lang_instruction,
            language=language,
            provider=provider,
            model=model,
            completion_fn=completion_fn,
        )
        llm_success = llm_success or conc_ok

        if not llm_success:
            return None  # every section degraded → let caller use derived/fallback

        # ── 4. Pure-code assembly (zero LLM calls) ──
        return self._assemble_report(
            query=query,
            target=target,
            exec_summary=exec_summary,
            narratives=narratives,
            market_analysis=market_analysis,
            swot_analysis=swot_analysis,
            conclusion=conclusion,
            sources=sources,
            profiles=profiles,
        )

    # ── Section generators ────────────────────────────────────────────

    async def _generate_executive_summary(
        self,
        target: str,
        market_analysis: dict[str, Any],
        swot_analysis: dict[str, Any],
        profiles: list[dict[str, Any]],
        review_feedback: dict[str, Any],
        sources: list[str],
        lang_instruction: str,
        language: str,
        provider: str,
        model: str,
        completion_fn: Any,
    ) -> tuple[str, bool]:
        """Generate a 2-3 sentence executive summary (output ~200 tokens).

        Returns ``(text, llm_ok)`` — *llm_ok* is True when the text came from
        the LLM (as opposed to a local fallback).
        """
        prompt = (
            f"{lang_instruction}\n"
            f"Write a 2-3 sentence executive summary for a competitive intelligence "
            f"report about {target}. Summarise the market landscape and key competitive "
            f"dynamics.  Do NOT use the word 'placeholder'.  Return plain text, no JSON.\n\n"
            f"Target: {target}\n"
            f"Market overview: {safe_json_dumps(market_analysis.get('market_overview', '')[:300])}\n"
            f"SWOT summary: {safe_json_dumps(swot_analysis.get('summary', '')[:300])}\n"
            f"Competitor count: {len(profiles)}\n"
            f"Prior review feedback: {safe_json_dumps(review_feedback)[:200]}\n"
        )
        try:
            raw = await completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=300,
                temperature=0.3,
                timeout=30.0,
            )
        except TypeError:
            raw = await completion_fn(prompt)
        except Exception as exc:
            logger.warning("Executive summary generation failed; using derived summary: %s", exc)
            return self._fallback_exec_summary(target, profiles, market_analysis, language=language), False

        text = self._extract_text_from_completion(
            raw,
            preferred_keys=("executive_summary", "summary"),
            section_title_keywords=("summary", "摘要", "执行摘要"),
        )
        if not text:
            return self._fallback_exec_summary(target, profiles, market_analysis, language=language), False
        return self._ensure_citation(text[:400], sources), True

    async def _generate_competitor_narratives(
        self,
        profiles: list[dict[str, Any]],
        sources: list[str],
        lang_instruction: str,
        language: str,
        provider: str,
        model: str,
        completion_fn: Any,
    ) -> list[dict[str, Any]]:
        """Generate a short narrative for each competitor — with concurrency control."""
        # Limit concurrent LLM calls to avoid overwhelming the API provider.
        # DeepSeek free tier typically allows 2-5 concurrent connections.
        _sem = asyncio.Semaphore(3)

        async def _narrative_for(profile: dict[str, Any]) -> dict[str, Any]:
            async with _sem:
                return await self._generate_competitor_narrative(
                    profile=profile,
                    sources=sources,
                    lang_instruction=lang_instruction,
                    language=language,
                    provider=provider,
                    model=model,
                    completion_fn=completion_fn,
                )

        tasks = [
            _narrative_for(p)
            for p in profiles
            if isinstance(p, dict) and p.get("name")
        ]
        if not tasks:
            return []

        results = await asyncio.gather(*tasks)
        return results

    async def _generate_competitor_narrative(
        self,
        profile: dict[str, Any],
        sources: list[str],
        lang_instruction: str,
        language: str,
        provider: str,
        model: str,
        completion_fn: Any,
    ) -> dict[str, Any]:
        """Generate a 2-3 paragraph narrative for a single competitor (output ~400 tokens)."""
        name = profile.get("name", "unknown")
        summary = str(profile.get("summary") or "")
        search_snippets = [
            str(s.get("snippet", ""))[:200]
            for s in (profile.get("search_results") or [])[:3]
            if isinstance(s, dict)
        ]
        rag_texts = [
            str(r.get("text", ""))[:200]
            for r in (profile.get("rag_context") or [])[:2]
            if isinstance(r, dict)
        ]
        competitor_sources = [
            s for s in (
                *[str(x.get("url", "")) for x in (profile.get("search_results") or []) if isinstance(x, dict)],
                *[str(x.get("source", "")) for x in (profile.get("rag_context") or []) if isinstance(x, dict)],
            )
            if s and s not in {"search_worker", "scrape_worker", "rag_retriever"}
        ][:3]

        prompt = (
            f"{lang_instruction}\n"
            f"Write 2-3 paragraphs analysing {name} as a competitor. "
            f"Cover what they do, their competitive position, and key differentiators. "
            f"Do NOT use the word 'placeholder'. Return plain markdown text, no JSON.\n\n"
            f"Profile: {safe_json_dumps(summary[:400])}\n"
            f"Search snippets: {safe_json_dumps(search_snippets)}\n"
            f"RAG context: {safe_json_dumps(rag_texts)}\n"
            f"Sources: {safe_json_dumps(competitor_sources)}\n"
        )
        try:
            raw = await completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=500,
                temperature=0.3,
                timeout=60.0,
            )
        except TypeError:
            raw = await completion_fn(prompt)
        except Exception as exc:
            logger.warning("Narrative generation failed for %s; using profile summary: %s", name, exc)
            content = summary or self._empty_profile_text(name, language)
            return {
                "name": name,
                "title": name,
                "content": self._ensure_citation(content, competitor_sources or sources),
                "key_insights": [summary] if summary else [content],
                "_llm": False,
            }

        text = self._extract_text_from_completion(
            raw,
            preferred_keys=("content", "analysis", "summary"),
            section_title_keywords=(name.lower(), "竞争格局", "competitor", "profile"),
        )
        insights = self._extract_insights_from_completion(raw, text)
        if not text:
            text = summary or self._empty_profile_text(name, language)
        return {
            "name": name,
            "title": name,
            "content": self._ensure_citation(text, competitor_sources or sources),
            "key_insights": insights or [text[:200]],
            "_llm": True,
        }

    async def _generate_conclusion(
        self,
        target: str,
        query: str,
        narratives: list[dict[str, Any]],
        swot_analysis: dict[str, Any],
        market_analysis: dict[str, Any],
        sources: list[str],
        lang_instruction: str,
        language: str,
        provider: str,
        model: str,
        completion_fn: Any,
    ) -> tuple[str, bool]:
        """Generate a short conclusion (output ~300 tokens).

        Returns ``(text, llm_ok)``.
        """
        narrative_summary = "\n".join(
            f"{n.get('name', '?')}: {str(n.get('content', ''))[:200]}"
            for n in narratives[:5]
        )
        prompt = (
            f"{lang_instruction}\n"
            f"Write a 2-3 sentence conclusion for a competitive intelligence report "
            f"about {target} in response to '{query}'. "
            f"Synthesise the key competitive dynamic and suggest 1-2 areas for "
            f"further investigation. Do NOT use the word 'placeholder'. "
            f"Return plain text, no JSON.\n\n"
            f"Competitor narratives:\n{narrative_summary}\n"
            f"SWOT cross analysis: {safe_json_dumps(swot_analysis.get('cross_analysis', {}))[:400]}\n"
            f"Market trends: {safe_json_dumps(market_analysis.get('growth_trends', []))[:300]}\n"
            f"Sources: {safe_json_dumps(sources[:5])}\n"
        )
        try:
            raw = await completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=400,
                temperature=0.3,
                timeout=30.0,
            )
        except TypeError:
            raw = await completion_fn(prompt)
        except Exception as exc:
            logger.warning("Conclusion generation failed; using derived conclusion: %s", exc)
            return (
                self._fallback_conclusion(target, language)
            ), False

        text = self._extract_text_from_completion(
            raw,
            preferred_keys=("conclusion", "recommendation", "summary"),
            section_title_keywords=("conclusion", "结论", "recommendation", "建议"),
        )
        if not text:
            return self._fallback_conclusion(target, language), False
        return self._ensure_citation(text, sources), True

    # ── Pure-code assembler ──────────────────────────────────────────

    def _assemble_report(
        self,
        query: str,
        target: str,
        exec_summary: str,
        narratives: list[dict[str, Any]],
        market_analysis: dict[str, Any],
        swot_analysis: dict[str, Any],
        conclusion: str,
        sources: list[str],
        profiles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the final report dict from independent sections.

        This is pure code — no LLM calls, no network, cannot fail.
        """
        # Competitor profile sections (from narratives)
        competitor_sections = [
            {
                "title": n.get("title", n.get("name", "Competitor")),
                "content": n.get("content", ""),
                "key_insights": self._normalize_list(n.get("key_insights", [])),
            }
            for n in narratives
        ]

        # Market analysis section (from upstream data, not LLM)
        market_section = {
            "title": "Market Analysis",
            "content": self._summarize_market(market_analysis),
            "key_insights": self._normalize_list(market_analysis.get("growth_trends", [])),
        }

        # SWOT section (from upstream data, not LLM)
        swot_section = {
            "title": "SWOT Analysis",
            "content": self._summarize_swot(swot_analysis),
            "key_insights": [swot_analysis.get("summary", "")] if swot_analysis.get("summary") else [],
        }

        sections = [*competitor_sections, market_section, swot_section]

        data_gaps: list[str] = []
        if not sources:
            data_gaps.append("No source URLs were available in competitor profiles.")
        if not profiles:
            data_gaps.append("No competitor profiles were available.")
        if not market_analysis:
            data_gaps.append("No market analysis was available.")
        if not swot_analysis:
            data_gaps.append("No SWOT analysis was available.")

        return {
            "title": f"CompIntel Report: {query}",
            "executive_summary": self._truncate_preserving_source(exec_summary, 300),
            "profiles": profiles,
            "market_analysis": market_analysis,
            "swot_analysis": swot_analysis,
            "sections": sections,
            "conclusion": self._truncate_preserving_source(conclusion, 400),
            "sources": sources,
            "data_gaps": data_gaps,
        }

    def _fallback_exec_summary(
        self,
        target: str,
        profiles: list[dict[str, Any]],
        market_analysis: dict[str, Any],
        language: str = "en",
    ) -> str:
        """Produce an executive summary from data without any LLM call."""
        competitor_names = [
            p.get("name", "") for p in profiles
            if isinstance(p, dict) and p.get("name")
        ]
        competitor_str = ", ".join(competitor_names[:4])
        if len(competitor_names) > 4:
            competitor_str += f" and {len(competitor_names) - 4} more"
        market_overview = str(market_analysis.get("market_overview", "")).strip()

        if language == "zh":
            base = f"本报告分析了 {target} 及其 {len(profiles)} 个竞品的竞争格局"
            if competitor_str:
                base += f"（含 {competitor_str}）"
            if market_overview:
                base += f"。{market_overview[:200]}"
            return f"{base}。"[:400]
        base = f"This report analyses the competitive landscape for {target} "
        base += f"across {len(profiles)} competitors"
        if competitor_str:
            base += f" including {competitor_str}"
        if market_overview:
            base += f". {market_overview[:200]}"
        return f"{base}."[:400]

    def _fallback_conclusion(self, target: str, language: str) -> str:
        msg = _FALLBACK_CONCLUSION.get(language, _FALLBACK_CONCLUSION["en"])
        return msg.format(target=target)[:400]

    @staticmethod
    def _empty_profile_text(name: str, language: str) -> str:
        if language == "zh":
            return f"{name} 是本次竞品研究中需要继续补充外部证据的竞争对象。"
        return f"{name} is a competitor that needs additional external evidence."

    def _extract_text_from_completion(
        self,
        raw: Any,
        preferred_keys: tuple[str, ...],
        section_title_keywords: tuple[str, ...] = (),
    ) -> str:
        """Turn either plain text or legacy JSON report output into section text."""
        text = str(raw).strip()
        parsed = load_repaired_json(text)
        if not isinstance(parsed, dict):
            return self._clean_completion_text(text)

        for key in preferred_keys:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return self._clean_completion_text(value)

        sections = parsed.get("sections")
        if isinstance(sections, list):
            selected = self._select_section(sections, section_title_keywords)
            if selected:
                return selected

        return self._clean_completion_text(text)

    def _extract_insights_from_completion(self, raw: Any, text: str) -> list[str]:
        parsed = load_repaired_json(str(raw).strip())
        if isinstance(parsed, dict):
            insights = self._normalize_list(parsed.get("key_insights", []))
            if insights:
                return insights[:3]
            sections = parsed.get("sections")
            if isinstance(sections, list):
                for section in sections:
                    if isinstance(section, dict):
                        insights = self._normalize_list(section.get("key_insights", []))
                        if insights:
                            return insights[:3]
        return [line.strip("- ") for line in text.split("\n") if line.strip().startswith("-")][:3]

    def _select_section(self, sections: list[Any], keywords: tuple[str, ...]) -> str:
        fallback = ""
        lowered_keywords = tuple(keyword.lower() for keyword in keywords if keyword)
        for section in sections:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title", "")).lower()
            content = str(section.get("content", "")).strip()
            if content and not fallback:
                fallback = content
            if content and lowered_keywords and any(keyword in title for keyword in lowered_keywords):
                return self._clean_completion_text(content)
        return self._clean_completion_text(fallback)

    @staticmethod
    def _clean_completion_text(text: str) -> str:
        cleaned = text.strip().strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        return cleaned

    @staticmethod
    def _ensure_citation(text: str, sources: list[str]) -> str:
        clean = text.strip()
        if not clean or "[Source:" in clean or not sources:
            return clean
        return f"{clean} [Source: {sources[0]}]"

    @staticmethod
    def _truncate_preserving_source(text: str, limit: int) -> str:
        clean = text.strip()
        if len(clean) <= limit:
            return clean
        source_start = clean.rfind("[Source:")
        if source_start == -1:
            return self._truncate_at_boundary(clean, limit)
        source = clean[source_start:].strip()
        room = max(0, limit - len(source) - 1)
        if room <= 0:
            return source[:limit]
        body = self._truncate_at_boundary(clean[:room], room)
        return f"{body.rstrip()} {source}"

    @staticmethod
    def _truncate_at_boundary(text: str, limit: int) -> str:
        """Truncate at a natural boundary (sentence end, space) instead of mid-word.

        P2-2: Prevents cutting English words or Chinese characters in half.
        Searches backwards from *limit* for a period / comma / space / CJK full-stop
        within the last 30 chars.  Falls back to hard truncation if no boundary found.
        """
        if len(text) <= limit:
            return text
        # Look backwards from limit for a natural break within 30 chars
        window = text[max(0, limit - 30):limit]
        # Sentence-ending markers (EN + CJK)
        breaks = {"。", ".", "！", "!", "；", ";", "\n", "，", ",", " "}
        best = limit
        for ch in reversed(window):
            if ch in breaks:
                best = text.index(ch, max(0, limit - 30)) + 1
                break
        return text[:best].rstrip()


    def _fallback_report(
        self,
        query: str,
        intent: dict[str, Any],
        profiles: list[dict[str, Any]],
        market_analysis: dict[str, Any],
        swot_analysis: dict[str, Any],
        language: str = "en",
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
        language: str = "en",
    ) -> dict[str, Any]:
        report = self._fallback_report(query, intent, profiles, market_analysis, swot_analysis, language)
        target = intent.get("target", "unknown")
        sources = report.get("sources", [])
        citation = f" [Source: {sources[0]}]" if sources else ""
        msg = _FALLBACK_EXECUTIVE_SUMMARY.get(language, _FALLBACK_EXECUTIVE_SUMMARY["en"])
        report["executive_summary"] = (msg.format(target=target))[:300]
        report["conclusion"] = self._build_derived_conclusion(target, profiles, sources, citation, language)
        report["data_gaps"] = [
            "Validate the latest revenue, customer count, and pricing details with authoritative sources.",
            *report.get("data_gaps", []),
        ]
        return report

    @staticmethod
    def _build_derived_conclusion(target: str, profiles: list[dict[str, Any]],
                                   sources: list[str], citation: str,
                                   language: str = "en") -> str:
        total_search = sum(len(p.get("search_results", [])) for p in profiles if isinstance(p, dict))
        total_scraped = sum(len(p.get("scraped_content", [])) for p in profiles if isinstance(p, dict))
        names = [p.get("name", "") for p in profiles if isinstance(p, dict) and p.get("name")]
        competitor_str = ", ".join(names[:3])
        connector = " 等" if language == "zh" else " and "
        if len(names) > 3:
            competitor_str += f"{connector}{len(names) - 3} more"
        if language == "zh":
            return (
                f"本分析基于 {total_search} 条搜索结果和 {total_scraped} 个抓取页面，"
                f"覆盖 {len(profiles)} 个竞争对手"
                f"{' (' + competitor_str + ')' if competitor_str else ''}。"
                f"建议通过 LLM 深度分析获取更精准的战略洞察。"
                f"{citation}"
            )[:400]
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
