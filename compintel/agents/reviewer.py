"""Review gate for CompIntel Research."""

from __future__ import annotations
import logging

from typing import Any

logger = logging.getLogger(__name__)

from ..parsing import safe_json_dumps
from ..prompts import load_prompt
from .base import BaseCompIntelAgent


class ReviewerAgent(BaseCompIntelAgent):
    MAX_RETRIES = 3

    def __init__(self, model: str = "deepseek-chat") -> None:
        super().__init__(model=model, model_key="strategic")

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        report = s.report
        retry_count = s.retry_count
        market_segment = s.market_segment

        feedback = await self._try_llm_review(report, retry_count)
        source = "llm"
        if feedback is None:
            feedback = self._fallback_review(report, retry_count, market_segment)
            source = "rules"

        approved = float(feedback.get("score", 0)) >= 7
        feedback["approved"] = approved
        feedback["retry_count"] = retry_count
        if approved:
            feedback["issues"] = []
        elif not feedback.get("issues"):
            feedback["issues"] = [{"type": "quality_gate", "detail": "report score is below approval threshold"}]
        feedback["note"] = feedback.get("note") or ("Approved" if approved else "Needs revision")
        detail = "approved" if approved else "revision requested"
        detail = f"{source}: {detail}"
        return {
            "review_feedback": feedback,
            "execution_log": [
                {"node": "reviewer", "event": "completed", "detail": detail}
            ],
        }

    async def _try_llm_review(
        self,
        report: dict[str, Any],
        retry_count: int,
    ) -> dict[str, Any] | None:
        prompt = load_prompt("reviewer")
        parsed = await self.llm.call_and_parse(
            prompt.format(retry_count=retry_count, report=safe_json_dumps(report)),
            model_key="smart",
            max_tokens=1200,
            temperature=prompt.temperature,
            max_attempts=3,
        )
        if isinstance(parsed, dict):
            return self._normalize_review(parsed, retry_count)
        return None

    def _normalize_review(self, payload: dict[str, Any], retry_count: int) -> dict[str, Any]:
        completeness = self._bounded_score(payload.get("completeness", 0))
        accuracy = self._bounded_score(payload.get("accuracy", 0))
        actionability = self._bounded_score(payload.get("actionability", 0))
        score = round(completeness * 0.4 + accuracy * 0.4 + actionability * 0.2, 1)
        feedback = {
            "score": score,
            "dimensions": {
                "completeness": completeness,
                "accuracy": accuracy,
                "actionability": actionability,
            },
            "approved": score >= 7,
            "retry_count": retry_count,
            "issues": self._normalize_issues(payload.get("issues", [])),
            "note": str(payload.get("note", "")),
        }
        return feedback

    _TEMPLATE_PATTERNS = (
        "Profile summary for",
        "placeholder",
        "Insufficient data to determine",
        "Insufficient profile data",
        "Insufficient search data",
        # P0-2: seed placeholder variants that indicate zero real data
        "Seed notes should be replaced",
        "GPT Researcher generated reports",
        "is a company tracked by CompIntel Research",
    )

    _INTERNAL_SOURCE_MARKERS = (
        "search_worker", "scrape_worker", "rag_retriever",
        "seed:", "tavily",
    )

    _SAAS_ONLY_KEYWORDS = (
        "workflow", "collaboration software", "knowledge management",
        "documents and databases", "team communication",
    )

    _EMPTY_DATA_PATTERNS = (
        "profiling skipped",
        "source unavailable",
        "No module named",
        "HTTP 403",
    )

    @staticmethod
    def _is_real_source(source: str) -> bool:
        """Return True only for real external URLs, not internal labels."""
        if source.startswith("http://") or source.startswith("https://"):
            return True
        return False

    def _fallback_review(self, report: dict[str, Any], retry_count: int,
                         market_segment: str = "") -> dict[str, Any]:
        sections = report.get("sections") if isinstance(report, dict) else []
        sources = report.get("sources") if isinstance(report, dict) else []
        profiles = report.get("profiles") if isinstance(report, dict) else []
        section_count = len(sections) if isinstance(sections, list) else 0
        source_count = len(sources) if isinstance(sources, list) else 0
        profile_count = len(profiles) if isinstance(profiles, list) else 0
        has_summary = bool(report.get("executive_summary")) if isinstance(report, dict) else False
        has_citations = "[Source:" in safe_json_dumps(sections)
        section_items = sections if isinstance(sections, list) else []
        has_insights = any(
            isinstance(section, dict) and section.get("key_insights")
            for section in section_items
        )
        has_conclusion = bool(report.get("conclusion")) if isinstance(report, dict) else False

        # ── Content quality checks ──
        report_text = safe_json_dumps(report).lower()

        # P0-2: Source quality — count REAL external URLs vs internal labels
        real_sources = [s for s in sources if self._is_real_source(str(s))]
        real_source_count = len(real_sources)
        all_sources_internal = source_count > 0 and real_source_count == 0

        # Template placeholder penalty: each hit costs 2 points from completeness
        template_hits = sum(1 for pat in self._TEMPLATE_PATTERNS if pat.lower() in report_text)
        template_penalty = template_hits * 2.0  # max 10 if 5+ hits

        # P0-2: Empty data detection — search / scrape / rag all failed
        empty_data_hits = sum(1 for pat in self._EMPTY_DATA_PATTERNS if pat.lower() in report_text)
        has_real_profiles = any(
            isinstance(p, dict) and p.get("summary")
            and "Seed notes should be replaced" not in str(p.get("summary", ""))
            and "GPT Researcher" not in str(p.get("summary", ""))
            for p in (profiles or [])
        )

        # Industry mismatch check: SaaS keywords in non-SaaS segment
        industry_mismatch = False
        if market_segment and not any(
            kw in market_segment for kw in ("协作", "collaboration", "SaaS", "软件", "知识管理",
                                             "文档", "生产力", "productivity", "笔记", "note")
        ):
            saas_hits = [kw for kw in self._SAAS_ONLY_KEYWORDS if kw in report_text]
            if len(saas_hits) >= 2:
                industry_mismatch = True

        completeness = 0.0
        if has_summary:
            completeness += 2.0
        completeness += min(section_count, 3) * 2.0
        if profile_count:
            completeness += 2.0
        completeness = max(0.0, completeness - template_penalty)
        completeness = min(completeness, 10.0)

        accuracy = 4.0
        # P0-2: real sources matter, not just any sources
        if real_source_count:
            accuracy += 2.0
        if has_citations:
            accuracy += 2.0
        if has_real_profiles:
            accuracy += 1.0
        # P0-2: catastrophic penalty when all sources are fake internal labels
        if all_sources_internal:
            accuracy = max(0.0, accuracy - 5.0)
        # P0-2: penalty for empty data channels
        if empty_data_hits >= 2:
            accuracy = max(0.0, accuracy - 3.0)
        if industry_mismatch:
            accuracy = max(0.0, accuracy - 4.0)
        accuracy = min(accuracy, 10.0)

        actionability = 4.0
        if has_insights:
            actionability += 3.0
        if has_conclusion:
            actionability += 2.0
        if report.get("data_gaps") is not None:
            actionability += 1.0
        actionability = min(actionability, 10.0)

        score = round(completeness * 0.4 + accuracy * 0.4 + actionability * 0.2, 1)
        issues = []
        if section_count < 3:
            issues.append({"type": "missing_sections", "detail": "report should include at least 3 structured sections"})
        if not source_count:
            issues.append({"type": "missing_sources", "detail": "report should include traceable sources"})
        if not has_citations:
            issues.append({"type": "missing_citations", "detail": "section content should include [Source: url] citations"})
        if not has_insights:
            issues.append({"type": "missing_insights", "detail": "sections should include key_insights"})
        if template_hits:
            issues.append({"type": "template_content", "detail": f"found {template_hits} template placeholder(s) in report content"})
        if industry_mismatch:
            issues.append({"type": "industry_mismatch", "detail": "market analysis may contain language from a different industry"})
        # P0-2: data quality issues
        if all_sources_internal:
            issues.append({"type": "no_real_sources", "detail": "all sources are internal labels (search_worker, seed:, etc.) — no real URLs were collected"})
        if empty_data_hits >= 2:
            issues.append({"type": "empty_data_channels", "detail": f"multiple data channels failed — search or scrape returned no real data ({empty_data_hits} failure indicators detected)"})
        if not has_real_profiles and profile_count > 0:
            issues.append({"type": "seed_profiles_only", "detail": "all competitor profiles are seed placeholders — no real competitive intelligence was collected"})

        # P0-2: hard cap when the report is clearly empty of real data
        if (template_hits >= 2 or all_sources_internal) and not has_real_profiles:
            score = min(score, 4.0)

        return {
            "score": score,
            "dimensions": {
                "completeness": completeness,
                "accuracy": accuracy,
                "actionability": actionability,
            },
            "approved": score >= 7,
            "retry_count": retry_count,
            "issues": issues,  # P0-2: always return issues so user can see what's wrong
            "note": "Approved" if score >= 7 else "Needs revision",
        }

    def _normalize_issues(self, values: Any) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, dict):
                issues.append(
                    {
                        "type": str(value.get("type", "quality")),
                        "detail": str(value.get("detail") or value.get("message") or value),
                    }
                )
            elif str(value).strip():
                issues.append({"type": "quality", "detail": str(value)})
        return issues

    def _bounded_score(self, value: Any) -> float:
        try:
            return max(0.0, min(10.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
