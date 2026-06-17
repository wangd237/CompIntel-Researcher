"""Workflow assembly for CompIntel Research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents.competitor_profiler import CompetitorProfilerAgent
from .agents.intent_analyst import IntentAnalystAgent
from .agents.market_analyst import MarketAnalystAgent
from .agents.report_writer import ReportWriterAgent
from .agents.research_planner import ResearchPlannerAgent
from .agents.reviewer import ReviewerAgent
from .agents.swot_synthesizer import SWOTSynthesizerAgent
from .schemas import CompIntelAnalyzeResponse, CompetitorProfileSchema


@dataclass(slots=True)
class GraphNode:
    name: str
    description: str


@dataclass(slots=True)
class CompIntelGraph:
    """Lightweight orchestration facade for Week 1 and Week 2."""

    model: str = "deepseek-chat"
    nodes: list[GraphNode] = field(default_factory=list)
    intent_analyst: IntentAnalystAgent = field(init=False, repr=False)
    research_planner: ResearchPlannerAgent = field(init=False, repr=False)
    competitor_profiler: CompetitorProfilerAgent = field(init=False, repr=False)
    market_analyst: MarketAnalystAgent = field(init=False, repr=False)
    swot_synthesizer: SWOTSynthesizerAgent = field(init=False, repr=False)
    report_writer: ReportWriterAgent = field(init=False, repr=False)
    reviewer: ReviewerAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.nodes:
            self.nodes = [
                GraphNode("intent_analyst", "Parse query into competitors and research questions"),
                GraphNode("research_planner", "Turn intent into analysis plan"),
                GraphNode("competitor_profiler", "Profile competitors in parallel"),
                GraphNode("market_analyst", "Aggregate market landscape"),
                GraphNode("swot_synthesizer", "Build SWOT matrix"),
                GraphNode("report_writer", "Write the final report"),
                GraphNode("reviewer", "Gate the report for quality"),
            ]
        self.intent_analyst = IntentAnalystAgent(model=self.model)
        self.research_planner = ResearchPlannerAgent(model=self.model)
        self.competitor_profiler = CompetitorProfilerAgent(model=self.model)
        self.market_analyst = MarketAnalystAgent(model=self.model)
        self.swot_synthesizer = SWOTSynthesizerAgent(model=self.model)
        self.report_writer = ReportWriterAgent(model=self.model)
        self.reviewer = ReviewerAgent(model=self.model)

    def describe(self) -> list[dict[str, str]]:
        return [{"name": node.name, "description": node.description} for node in self.nodes]

    async def run_intent_only(self, query: str) -> dict[str, Any]:
        return await self.intent_analyst(query)

    async def run_competitor_pipeline(self, query: str) -> CompIntelAnalyzeResponse:
        intent_result = await self.intent_analyst(query)
        plan_result = await self.research_planner(intent_result)

        profiles: list[CompetitorProfileSchema] = []
        for competitor in intent_result.get("competitors", []):
            profile_result = await self.competitor_profiler(
                {
                    "competitor": competitor,
                    "research_questions": intent_result.get("research_questions", []),
                }
            )
            profiles.append(CompetitorProfileSchema(**profile_result["profile"]))

        market_result = await self.market_analyst(
            {
                "profiles": [profile.model_dump() for profile in profiles],
                "market_segment": intent_result.get("market_segment"),
            }
        )
        swot_result = await self.swot_synthesizer(
            {
                "profiles": [profile.model_dump() for profile in profiles],
                "market_analysis": market_result.get("market_analysis", {}),
            }
        )
        report_result = await self.report_writer(
            {
                "query": query,
                "intent": intent_result.get("intent", {}),
                "profiles": [profile.model_dump() for profile in profiles],
                "market_analysis": market_result.get("market_analysis", {}),
                "swot_analysis": swot_result.get("swot_analysis", {}),
            }
        )
        review_result = await self.reviewer(
            {
                "report": report_result.get("report", {}),
                "review_feedback": {},
            }
        )

        return CompIntelAnalyzeResponse(
            query=query,
            intent=intent_result.get("intent"),
            competitors=intent_result.get("competitors", []),
            profiles=profiles,
            report={
                "research_plan": plan_result.get("research_plan", {}),
                "market_analysis": market_result.get("market_analysis", {}),
                "swot_analysis": swot_result.get("swot_analysis", {}),
                "report": report_result.get("report", {}),
                "review_feedback": review_result.get("review_feedback", {}),
                "execution_log": [
                    *plan_result.get("execution_log", []),
                    *market_result.get("execution_log", []),
                    *swot_result.get("execution_log", []),
                    *report_result.get("execution_log", []),
                    *review_result.get("execution_log", []),
                ],
            },
            warnings=intent_result.get("notes", []),
        )

    def describe_pipeline(self) -> dict[str, Any]:
        return {
            "entrypoint": "intent_analyst",
            "stages": [
                "intent_analyst",
                "research_planner",
                "competitor_profiler",
                "market_analyst",
                "swot_synthesizer",
                "report_writer",
                "reviewer",
            ],
            "current_capacity": "intent -> plan -> profiles",
        }
