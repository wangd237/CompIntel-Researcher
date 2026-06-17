"""Review gate for CompIntel Research."""

from __future__ import annotations

from typing import Any

from ..llm import _split_provider_model
from ..parsing import load_repaired_json, safe_json_dumps
from ..settings import CompIntelSettings
from .base import BaseCompIntelAgent


class ReviewerAgent(BaseCompIntelAgent):
    MAX_RETRIES = 3

    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model)
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        report = {}
        retry_count = 0
        if isinstance(state, dict):
            report = state.get("report") or {}
            retry_count = int((state.get("review_feedback") or {}).get("retry_count", 0))

        settings = CompIntelSettings.from_env()
        feedback = await self._try_llm_review(report, retry_count, settings)
        source = "llm"
        if feedback is None:
            feedback = self._fallback_review(report, retry_count)
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
        settings: CompIntelSettings,
    ) -> dict[str, Any] | None:
        if not settings.llm_api_key and self.completion_fn is None:
            return None

        completion_fn = self.completion_fn
        if completion_fn is None:
            try:
                from ..llm import create_chat_completion
            except Exception:
                return None
            completion_fn = create_chat_completion

        provider, model = _split_provider_model(settings.strategic_llm)
        prompt = (
            "You are CompIntel's LLM-as-Judge reviewer. Score the report on three "
            "dimensions from 0 to 10: completeness, accuracy, actionability. "
            "Return strict JSON with keys completeness, accuracy, actionability, "
            "issues, note. The final score will be completeness*0.4 + accuracy*0.4 "
            "+ actionability*0.2. Issues must be concrete rewrite instructions when "
            "a score is below 7.\n"
            f"Retry count: {retry_count}\n"
            f"Report: {safe_json_dumps(report)}\n"
        )
        try:
            raw = await completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=1200,
                temperature=0.1,
            )
        except TypeError:
            raw = await completion_fn(prompt)
        except Exception:
            return None

        parsed = load_repaired_json(str(raw))
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

    def _fallback_review(self, report: dict[str, Any], retry_count: int) -> dict[str, Any]:
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

        completeness = 0.0
        if has_summary:
            completeness += 2.0
        completeness += min(section_count, 3) * 2.0
        if profile_count:
            completeness += 2.0
        completeness = min(completeness, 10.0)

        accuracy = 4.0
        if source_count:
            accuracy += 2.0
        if has_citations:
            accuracy += 2.0
        if profile_count:
            accuracy += 1.0
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

        return {
            "score": score,
            "dimensions": {
                "completeness": completeness,
                "accuracy": accuracy,
                "actionability": actionability,
            },
            "approved": score >= 7,
            "retry_count": retry_count,
            "issues": [] if score >= 7 else issues,
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
