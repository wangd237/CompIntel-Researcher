"""Research planning for competitor profiling."""

from __future__ import annotations
import logging

from typing import Any

logger = logging.getLogger(__name__)

from ..parsing import safe_json_dumps
from ..prompts import load_prompt
from .base import BaseCompIntelAgent


class ResearchPlannerAgent(BaseCompIntelAgent):
    """Create a structured plan for each competitor."""

    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model, model_key="fast")
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        s = self.read_state(state)
        competitors = s.competitors
        questions = s.research_questions
        market_segment = s.market_segment

        llm_plan = await self._try_llm_plan(competitors, questions, market_segment)
        if llm_plan:
            return {
                "research_plan": llm_plan,
                "execution_log": [
                    {"node": "research_planner", "event": "completed", "detail": "llm competitor plan generated"}
                ],
            }

        plan = self._template_plan(competitors)
        return {
            "research_plan": plan,
            "execution_log": [
                {"node": "research_planner", "event": "completed", "detail": "template competitor plan generated"}
            ],
        }

    async def _try_llm_plan(
        self,
        competitors: list[dict[str, Any]],
        questions: list[str],
        market_segment: str,
    ) -> dict[str, Any] | None:
        if self.completion_fn is not None:
            return await self._legacy_llm_plan(competitors, questions, market_segment)

        prompt = load_prompt("research_planner")
        result = await self.llm.call_and_parse(
            prompt.format(
                market_segment=market_segment,
                competitors=safe_json_dumps(competitors),
                research_questions=safe_json_dumps(questions),
            ),
            model_key=prompt.model_key,
            max_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
        )
        if isinstance(result, dict):
            return result
        logger.warning("Research planner LLM call failed; using template plan")
        return None

    async def _legacy_llm_plan(
        self,
        competitors: list[dict[str, Any]],
        questions: list[str],
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

        provider, model = _split_provider_model(settings.fast_llm)
        prompt = (
            "You are CompIntel's research planner.\n"
            "Create a JSON object keyed by competitor name. Each competitor must include phases "
            "with focused search queries and a search_strategy.\n"
            "Return strict JSON only.\n"
            f"Market segment: {market_segment}\n"
            f"Competitors: {safe_json_dumps(competitors)}\n"
            f"Research questions: {safe_json_dumps(questions)}\n"
        )
        try:
            raw = await self.completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=1600,
                temperature=0.2,
            )
        except TypeError:
            raw = await self.completion_fn(prompt)
        except Exception as exc:
            logger.warning("Research planner LLM call failed; using template plan: %s", exc)
            return None

        from ..parsing import load_repaired_json
        parsed = load_repaired_json(str(raw))
        if isinstance(parsed, dict):
            return parsed
        return None

    def _template_plan(self, competitors: list[dict[str, Any]]) -> dict[str, Any]:
        plan: dict[str, Any] = {}
        for competitor in competitors:
            name = competitor.get("name", "unknown")
            plan[name] = {
                "phases": [
                    {"phase": "company_overview", "queries": self._overview_queries(name)},
                    {"phase": "product_analysis", "queries": self._product_queries(name)},
                    {"phase": "pricing", "queries": self._pricing_queries(name)},
                    {"phase": "market_position", "queries": self._market_queries(name)},
                ],
                "search_strategy": {
                    "sources": ["official_website", "tech_media", "review_platforms", "financial_db"],
                    "max_results_per_phase": 5,
                },
            }
        return plan

    def _overview_queries(self, name: str) -> list[str]:
        return [f"{name} company overview", f"{name} founding team", f"{name} mission"]

    def _product_queries(self, name: str) -> list[str]:
        return [f"{name} product features", f"{name} integrations", f"{name} use cases"]

    def _pricing_queries(self, name: str) -> list[str]:
        return [f"{name} pricing", f"{name} plans", f"{name} free tier"]

    def _market_queries(self, name: str) -> list[str]:
        return [f"{name} competitors", f"{name} market position", f"{name} reviews"]
