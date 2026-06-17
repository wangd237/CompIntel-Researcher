"""Research planning for competitor profiling."""

from __future__ import annotations

from typing import Any

from ..parsing import load_repaired_json, safe_json_dumps
from ..settings import CompIntelSettings
from .base import BaseCompIntelAgent


class ResearchPlannerAgent(BaseCompIntelAgent):
    """Create a structured plan for each competitor."""

    def __init__(self, model: str = "deepseek-chat", completion_fn: Any | None = None) -> None:
        super().__init__(model=model)
        self.completion_fn = completion_fn

    async def __call__(self, state: Any) -> dict[str, Any]:
        competitors = []
        questions: list[str] = []
        market_segment = "unknown"
        if isinstance(state, dict):
            competitors = state.get("competitors") or []
            questions = state.get("research_questions") or []
            market_segment = state.get("market_segment") or (state.get("intent") or {}).get("market_segment") or market_segment

        settings = CompIntelSettings.from_env()
        llm_plan = await self._try_llm_plan(competitors, questions, market_segment, settings)
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
        settings: CompIntelSettings,
    ) -> dict[str, Any] | None:
        if not settings.llm_api_key and self.completion_fn is None:
            return None

        completion_fn = self.completion_fn
        if completion_fn is None:
            try:
                from gpt_researcher.utils.llm import create_chat_completion
            except Exception:
                return None
            completion_fn = create_chat_completion

        provider, model = self._split_provider_model(settings.fast_llm)
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
            raw = await completion_fn(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                llm_provider=provider,
                max_tokens=1600,
                temperature=0.2,
            )
        except TypeError:
            raw = await completion_fn(prompt)
        except Exception:
            return None

        parsed = load_repaired_json(str(raw))
        if isinstance(parsed, dict):
            return parsed
        return None

    def _split_provider_model(self, value: str) -> tuple[str, str]:
        if ":" in value:
            provider, model = value.split(":", 1)
            return provider.strip() or "openai", model.strip() or "gpt-4o-mini"
        return "openai", value.strip() or "gpt-4o-mini"

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
