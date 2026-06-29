"""LangGraph workflow assembly for CompIntel Research."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents.competitor_profiler import CompetitorProfilerAgent
from .agents.curator import CuratorAgent
from .agents.editor import EditorAgent
from .agents.intent_analyst import IntentAnalystAgent
from .agents.market_analyst import MarketAnalystAgent
from .agents.report_writer import ReportWriterAgent
from .agents.research_planner import ResearchPlannerAgent
from .agents.reviewer import ReviewerAgent
from .agents.swot_synthesizer import SWOTSynthesizerAgent
from .schemas import CompIntelAnalyzeResponse, CompetitorProfileSchema
from .state import CompIntelState


# ── pipeline stage descriptors ──────────────────────────────────────────

_PIPELINE_STAGES: list[dict[str, str]] = [
    {"name": "intent_analyst",      "description": "Parse query into competitors and research questions"},
    {"name": "research_planner",    "description": "Turn intent into analysis plan"},
    {"name": "competitor_profiler", "description": "Profile competitors via fan-out"},
    {"name": "curator",             "description": "Clean profiles and grade evidence quality"},
    {"name": "market_analyst",      "description": "Aggregate market landscape"},
    {"name": "swot_synthesizer",    "description": "Build SWOT matrix (per competitor)"},
    {"name": "report_writer",       "description": "Write the final report"},
    {"name": "editor",              "description": "Editorial pass: unify voice, resolve conflicts"},
    {"name": "reviewer",            "description": "Gate the report for quality"},
    {"name": "rag_ingest",          "description": "Write approved report back into RAG memory"},
]


@dataclass(slots=True)
class CompIntelGraph:
    """LangGraph-backed orchestration facade for CompIntel Research."""

    model: str = "deepseek-chat"
    nodes: list[dict[str, str]] = field(default_factory=lambda: list(_PIPELINE_STAGES))
    checkpoint_path: str = "compintel_checkpoints.db"
    checkpointer: Any = field(default_factory=MemorySaver)
    intent_analyst: IntentAnalystAgent = field(init=False, repr=False)
    research_planner: ResearchPlannerAgent = field(init=False, repr=False)
    competitor_profiler: CompetitorProfilerAgent = field(init=False, repr=False)
    market_analyst: MarketAnalystAgent = field(init=False, repr=False)
    swot_synthesizer: SWOTSynthesizerAgent = field(init=False, repr=False)
    report_writer: ReportWriterAgent = field(init=False, repr=False)
    reviewer: ReviewerAgent = field(init=False, repr=False)
    curator: CuratorAgent = field(init=False, repr=False)
    editor: EditorAgent = field(init=False, repr=False)
    app: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.checkpoint_path, str) and isinstance(self.checkpointer, MemorySaver):
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-untyped]
                self.checkpointer = SqliteSaver.from_conn_string(self.checkpoint_path)
            except ImportError:
                pass

        self.intent_analyst = IntentAnalystAgent(model=self.model)
        self.research_planner = ResearchPlannerAgent(model=self.model)
        self.competitor_profiler = CompetitorProfilerAgent(model=self.model)
        self.market_analyst = MarketAnalystAgent(model=self.model)
        self.swot_synthesizer = SWOTSynthesizerAgent(model=self.model)
        self.report_writer = ReportWriterAgent(model=self.model)
        self.reviewer = ReviewerAgent(model=self.model)
        self.curator = CuratorAgent(model=self.model)
        self.editor = EditorAgent(model=self.model)
        self.app = self._build_graph()

    def describe(self) -> list[dict[str, str]]:
        return list(self.nodes)

    @staticmethod
    def _detect_language(query: str) -> str:
        chinese_chars = sum(1 for c in query if '一' <= c <= '鿯')
        return "zh" if chinese_chars >= 2 else "en"

    # ── public API ───────────────────────────────────────────────────────

    async def run_intent_only(self, query: str) -> dict[str, Any]:
        return await self.intent_analyst(query)

    async def run_competitor_pipeline(self, query: str) -> CompIntelAnalyzeResponse:
        state = await self.app.ainvoke(
            {"query": query, "profiles": [], "execution_log": [], "retry_count": 0},
            self._config(query),
        )
        profiles = [CompetitorProfileSchema(**p) for p in state.get("profiles", [])]
        return CompIntelAnalyzeResponse(
            query=query,
            intent=state.get("intent"),
            competitors=state.get("competitors", []),
            profiles=profiles,
            report={
                "research_plan": state.get("research_plan", {}),
                "market_analysis": state.get("market_analysis", {}),
                "swot_analysis": state.get("swot_analysis", {}),
                "report": state.get("report", {}),
                "review_feedback": state.get("review_feedback", {}),
                "execution_log": state.get("execution_log", []),
                "curator_evidence": state.get("curator_evidence", {}),
            },
            warnings=state.get("warnings", []),
        )

    def describe_pipeline(self) -> dict[str, Any]:
        return {
            "entrypoint": "intent_analyst",
            "stages": [n["name"] for n in self.nodes],
            "checkpointer": type(self.checkpointer).__name__,
        }

    def export_mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()

    def get_checkpoint(self, query: str) -> Any:
        return self.checkpointer.get_tuple(self._config(query))

    # ── graph assembly ───────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        graph = StateGraph(CompIntelState)
        graph.add_node("intent_analyst", self._intent_node)
        graph.add_node("research_planner", self._planner_node)
        graph.add_node("competitor_profiler", self._profile_one_node)
        graph.add_node("curator", self._curator_node)
        graph.add_node("market_analyst", self._market_node)
        graph.add_node("swot_synthesizer", self._swot_node)
        graph.add_node("report_writer", self._report_node)
        graph.add_node("editor", self._editor_node)
        graph.add_node("reviewer", self._review_node)
        graph.add_node("rag_ingest", self._rag_ingest_node)

        graph.add_edge(START, "intent_analyst")
        graph.add_edge("intent_analyst", "research_planner")
        graph.add_conditional_edges("research_planner", self._fan_out_competitors)
        graph.add_edge("competitor_profiler", "curator")
        graph.add_edge("curator", "market_analyst")
        graph.add_edge("market_analyst", "swot_synthesizer")
        graph.add_edge("swot_synthesizer", "report_writer")
        graph.add_edge("report_writer", "editor")
        graph.add_edge("editor", "reviewer")
        graph.add_conditional_edges(
            "reviewer", self._review_route,
            {"approved": "rag_ingest", "revise": "report_writer"},
        )
        graph.add_edge("rag_ingest", END)
        return graph.compile(checkpointer=self.checkpointer)

    # ── node callbacks ───────────────────────────────────────────────────

    async def _intent_node(self, state: CompIntelState) -> dict[str, Any]:
        query = state.get("query", "")
        result = await self.intent_analyst(query)
        intent = result.get("intent") or {}
        lang = self._detect_language(query)
        return {
            "language": lang,
            "intent": intent,
            "target": result.get("target") or intent.get("target"),
            "market_segment": result.get("market_segment") or intent.get("market_segment"),
            "competitors": result.get("competitors", []),
            "research_questions": result.get("research_questions", []),
            "warnings": result.get("notes", []),
            "execution_log": [{"node": "intent_analyst", "event": "completed", "detail": result.get("target", "unknown")}],
        }

    async def _planner_node(self, state: CompIntelState) -> dict[str, Any]:
        return await self.research_planner(state)

    def _fan_out_competitors(self, state: CompIntelState) -> list[Send] | Literal["market_analyst"]:
        competitors = state.get("competitors", [])
        if not competitors:
            return "market_analyst"
        return [
            Send("competitor_profiler", {
                "competitor": c,
                "research_questions": state.get("research_questions", []),
                "market_segment": state.get("market_segment", ""),
            })
            for c in competitors
        ]

    async def _profile_one_node(self, state: CompIntelState) -> dict[str, Any]:
        """Profile one competitor via CompetitorProfilerAgent (asyncio.gather, no subgraph)."""
        try:
            return await self.competitor_profiler(state)
        except Exception as exc:
            competitor = state.get("competitor") or {}
            name = competitor.get("name", "unknown")
            return {
                "profiles": [{"name": name, "summary": f"{name} profiling skipped — source unavailable"}],
                "warnings": [f"{name} profiling skipped — source unavailable."],
                "execution_log": [{"node": "competitor_profiler", "event": "error", "detail": str(exc)}],
            }

    # ── downstream nodes (thin dispatch to agents) ────────────────────────

    async def _curator_node(self, state: CompIntelState) -> dict[str, Any]:
        return await self.curator({
            "profiles": state.get("profiles", []),
            "market_segment": state.get("market_segment"),
            "language": state.get("language", "en"),
        })

    async def _market_node(self, state: CompIntelState) -> dict[str, Any]:
        profiles = state.get("curated_profiles") or state.get("profiles", [])
        return await self.market_analyst({
            "profiles": profiles,
            "market_segment": state.get("market_segment"),
            "language": state.get("language", "en"),
        })

    async def _swot_node(self, state: CompIntelState) -> dict[str, Any]:
        profiles = state.get("curated_profiles") or state.get("profiles", [])
        return await self.swot_synthesizer({
            "profiles": profiles,
            "market_analysis": state.get("market_analysis", {}),
            "language": state.get("language", "en"),
        })

    async def _report_node(self, state: CompIntelState) -> dict[str, Any]:
        profiles = state.get("curated_profiles") or state.get("profiles", [])
        return await self.report_writer({
            "query": state.get("query"),
            "intent": state.get("intent", {}),
            "profiles": profiles,
            "market_analysis": state.get("market_analysis", {}),
            "swot_analysis": state.get("swot_analysis", {}),
            "review_feedback": state.get("review_feedback", {}),
            "language": state.get("language", "en"),
        })

    async def _editor_node(self, state: CompIntelState) -> dict[str, Any]:
        return await self.editor({
            "report": state.get("report", {}),
            "language": state.get("language", "en"),
        })

    async def _review_node(self, state: CompIntelState) -> dict[str, Any]:
        result = await self.reviewer({
            "report": state.get("report", {}),
            "review_feedback": state.get("review_feedback", {}),
            "retry_count": state.get("retry_count", 0),
        })
        feedback = result.get("review_feedback", {})
        retry_count = int(state.get("retry_count", 0))
        if not feedback.get("approved"):
            retry_count += 1
        if retry_count >= ReviewerAgent.MAX_RETRIES and not feedback.get("approved"):
            lang = state.get("language", "en")
            feedback["review_banner"] = (
                "本报告在 3 次修订尝试后自动通过。建议由人工分析师审阅以用于关键决策。"
                if lang == "zh"
                else "This report was auto-approved after 3 revision attempts. Review by a human analyst is recommended for critical decisions."
            )
        return {
            "review_feedback": feedback,
            "retry_count": retry_count,
            "execution_log": result.get("execution_log", []),
        }

    def _review_route(self, state: CompIntelState) -> Literal["approved", "revise"]:
        feedback = state.get("review_feedback", {})
        if feedback.get("approved") or int(state.get("retry_count", 0)) >= ReviewerAgent.MAX_RETRIES:
            return "approved"
        return "revise"

    # ── RAG ingest ───────────────────────────────────────────────────────

    async def _rag_ingest_node(self, state: CompIntelState) -> dict[str, Any]:
        target = state.get("target") or state.get("intent", {}).get("target", "unknown")
        market_segment = state.get("market_segment", "")
        report = state.get("report", {})
        swot = state.get("swot_analysis", {})
        market = state.get("market_analysis", {})

        feedback = state.get("review_feedback", {})
        retry_count = int(state.get("retry_count", 0))
        is_auto_approved = retry_count >= ReviewerAgent.MAX_RETRIES and not feedback.get("approved")
        if not feedback.get("approved") and not is_auto_approved:
            return {"execution_log": [{"node": "rag_ingest", "event": "completed", "detail": "Skipped — report not approved"}]}

        quality = "limited" if is_auto_approved else "reviewed"
        documents = []
        now = datetime.now(timezone.utc).isoformat()

        exec_summary = report.get("executive_summary", "")
        conclusion = report.get("conclusion", "")
        if exec_summary or conclusion:
            documents.append({
                "text": f"Target: {target}\nMarket: {market_segment}\nExecutive Summary: {exec_summary}\nConclusion: {conclusion}",
                "source": f"report:{target}:{now}",
                "metadata": {"report_type": "executive_summary", "target": target, "market_segment": market_segment, "ingested_at": now, "quality": quality},
            })

        for comp in (swot.get("competitors") or []):
            if isinstance(comp, dict) and comp.get("name"):
                documents.append({
                    "text": f"SWOT for {comp.get('name')}: {comp}",
                    "source": f"swot:{comp.get('name')}:{now}",
                    "metadata": {"report_type": "swot", "target": target, "competitor": comp.get("name"), "market_segment": market_segment, "ingested_at": now, "quality": quality},
                })

        if market:
            documents.append({
                "text": f"Market analysis for {market_segment}: {market}",
                "source": f"market:{target}:{now}",
                "metadata": {"report_type": "market_analysis", "target": target, "market_segment": market_segment, "ingested_at": now, "quality": quality},
            })

        if documents:
            try:
                ingested = self.competitor_profiler.rag_retriever.store.ingest(documents)
                return {"execution_log": [{"node": "rag_ingest", "event": "completed", "detail": f"Ingested {ingested} chunks into RAG memory for {target}"}]}
            except Exception as exc:
                return {"execution_log": [{"node": "rag_ingest", "event": "completed_with_error", "detail": f"RAG ingest failed (non-fatal): {exc}"}]}
        return {"execution_log": [{"node": "rag_ingest", "event": "completed", "detail": "no documents to ingest"}]}

    # ── helpers ──────────────────────────────────────────────────────────

    def _config(self, query: str) -> dict[str, Any]:
        thread_id = f"compintel:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        return {"configurable": {"thread_id": thread_id}}
