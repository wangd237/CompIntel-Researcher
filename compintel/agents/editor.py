"""Editor-in-chief for CompIntel Research.

The final node before reviewer: takes the complete assembled report draft,
unifies terminology across sections, detects contradictions, deduplicates,
and rewrites the executive summary last (after seeing all chapters).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

from ..parsing import safe_json_dumps
from ..prompts import load_prompt
from .base import BaseCompIntelAgent


class EditorAgent(BaseCompIntelAgent):
    """Final editorial pass: unify voice, remove duplication, resolve conflicts.

    This agent runs AFTER report_writer has assembled a complete draft and
    BEFORE reviewer gates the report.  It has access to the full report
    and can rewrite the executive summary based on actual section content.
    """

    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model, model_key="smart")
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        report = s.report
        language = s.language

        if not isinstance(report, dict) or not report:
            return {"execution_log": [
                {"node": "editor", "event": "completed", "detail": "empty report, skipped"}
            ]}

        edited = await self._edit_report(report, language)
        source = "editor"
        if edited is None:
            edited = report  # pass through unchanged on editor failure
            source = "editor-pass-through"

        return {
            "report": edited,
            "execution_log": [
                {"node": "editor", "event": "completed",
                 "detail": f"{source}: {len(edited.get('sections', []))} sections reviewed"}
            ],
        }

    async def _edit_report(
        self, report: dict[str, Any], language: str
    ) -> dict[str, Any] | None:
        if self.completion_fn is not None:
            return None  # skip LLM editor for test-injected paths

        # Build a compact version: sections with titles + content,
        # plus the existing executive_summary and conclusion for context.
        compact = {
            "executive_summary": str(report.get("executive_summary", ""))[:400],
            "sections": [
                {
                    "title": s.get("title", ""),
                    "content": str(s.get("content", ""))[:600],
                    "key_insights": s.get("key_insights", []),
                }
                for s in (report.get("sections") or [])
                if isinstance(s, dict)
            ],
            "conclusion": str(report.get("conclusion", ""))[:400],
            "data_gaps": report.get("data_gaps", []),
        }

        lang_instr = (
            "用中文输出。"
            if language == "zh"
            else "Output in English."
        )

        try:
            prompt = load_prompt("editor")
            parsed = await self.llm.call_and_parse(
                prompt.format(
                    language_instruction=lang_instr,
                    draft=safe_json_dumps(compact),
                ),
                model_key=prompt.model_key,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
            )
            if isinstance(parsed, dict):
                result = dict(report)  # preserve unmentioned fields
                result["executive_summary"] = parsed.get("executive_summary", report.get("executive_summary", ""))
                result["conclusion"] = parsed.get("conclusion", report.get("conclusion", ""))
                result["data_gaps"] = parsed.get("data_gaps", report.get("data_gaps", []))
                if parsed.get("sections"):
                    result["sections"] = parsed["sections"]
                result["editor_notes"] = parsed.get("editor_notes", [])
                return result
        except Exception as exc:
            logger.warning("Editor failed (non-fatal): %s — passing report through unchanged", str(exc)[:200])
        return None
