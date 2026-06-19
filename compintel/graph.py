"""LangGraph workflow assembly for CompIntel Research."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from .agents.competitor_profiler import CompetitorProfilerAgent
from .agents.intent_analyst import IntentAnalystAgent
from .agents.market_analyst import MarketAnalystAgent
from .agents.report_writer import ReportWriterAgent
from .agents.research_planner import ResearchPlannerAgent
from .agents.reviewer import ReviewerAgent
from .agents.swot_synthesizer import SWOTSynthesizerAgent
from .schemas import CompIntelAnalyzeResponse, CompetitorProfileSchema
from .state import CompIntelState


class CompetitorProfileGraphState(TypedDict, total=False):
    competitor: dict[str, Any]
    research_questions: list[str]
    market_segment: str
    search_results: dict[str, Any]
    scraped_content: dict[str, Any]
    rag_context: dict[str, Any]
    profile: dict[str, Any]
    execution_log: Annotated[list[dict[str, Any]], operator.add]


@dataclass(slots=True)
class GraphNode:
    name: str
    description: str


@dataclass(slots=True)
class CompIntelGraph:
    """LangGraph-backed orchestration facade for CompIntel Research."""

    model: str = "deepseek-chat"
    nodes: list[GraphNode] = field(default_factory=list)
    checkpoint_path: str = "compintel_checkpoints.db"
    checkpointer: Any = field(default_factory=MemorySaver)
    intent_analyst: IntentAnalystAgent = field(init=False, repr=False)
    research_planner: ResearchPlannerAgent = field(init=False, repr=False)
    competitor_profiler: CompetitorProfilerAgent = field(init=False, repr=False)
    market_analyst: MarketAnalystAgent = field(init=False, repr=False)
    swot_synthesizer: SWOTSynthesizerAgent = field(init=False, repr=False)
    report_writer: ReportWriterAgent = field(init=False, repr=False)
    reviewer: ReviewerAgent = field(init=False, repr=False)
    app: Any = field(init=False, repr=False)
    profile_app: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Lazy SqliteSaver: if checkpoint_path is a string path, replace
        # the default MemorySaver with a persistent SqliteSaver.
        if isinstance(self.checkpoint_path, str) and isinstance(self.checkpointer, MemorySaver):
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-untyped]

                self.checkpointer = SqliteSaver.from_conn_string(self.checkpoint_path)
            except ImportError:
                pass  # SQLite support not installed; keep MemorySaver

        if not self.nodes:
            self.nodes = [
                GraphNode("intent_analyst", "Parse query into competitors and research questions"),
                GraphNode("research_planner", "Turn intent into analysis plan"),
                GraphNode("competitor_profiler", "Profile competitors via fan-out"),
                GraphNode("market_analyst", "Aggregate market landscape"),
                GraphNode("swot_synthesizer", "Build SWOT matrix"),
                GraphNode("report_writer", "Write the final report"),
                GraphNode("reviewer", "Gate the report for quality"),
                GraphNode("rag_ingest", "Write approved report back into RAG memory"),
            ]
        self.intent_analyst = IntentAnalystAgent(model=self.model)
        self.research_planner = ResearchPlannerAgent(model=self.model)
        self.competitor_profiler = CompetitorProfilerAgent(model=self.model)
        self.market_analyst = MarketAnalystAgent(model=self.model)
        self.swot_synthesizer = SWOTSynthesizerAgent(model=self.model)
        self.report_writer = ReportWriterAgent(model=self.model)
        self.reviewer = ReviewerAgent(model=self.model)
        self._bootstrap_rag_seeds()
        self.profile_app = self._build_profile_graph()
        self.app = self._build_graph()

    def _bootstrap_rag_seeds(self) -> None:
        """Load bootstrap seed reports into Qdrant so RAG has initial data."""
        try:
            from .rag import load_seed_reports
            store = self.competitor_profiler.rag_retriever.store
            if store.client is None or store.client.get_collections() is None:
                return
            collection_names = {c.name for c in store.client.get_collections().collections}
            if store.collection_name not in collection_names:
                load_seed_reports(store)
        except Exception:
            pass  # bootstrap is optional; RAG works with historical reports alone

    def describe(self) -> list[dict[str, str]]:
        return [{"name": node.name, "description": node.description} for node in self.nodes]

    async def run_intent_only(self, query: str) -> dict[str, Any]:
        return await self.intent_analyst(query)

    async def run_competitor_pipeline(self, query: str) -> CompIntelAnalyzeResponse:
        state = await self.app.ainvoke(
            {
                "query": query,
                "profiles": [],
                "execution_log": [],
                "retry_count": 0,
            },
            self._config(query),
        )
        profiles = [
            CompetitorProfileSchema(**profile)
            for profile in state.get("profiles", [])
        ]
        intent = state.get("intent")
        return CompIntelAnalyzeResponse(
            query=query,
            intent=intent,
            competitors=state.get("competitors", []),
            profiles=profiles,
            report={
                "research_plan": state.get("research_plan", {}),
                "market_analysis": state.get("market_analysis", {}),
                "swot_analysis": state.get("swot_analysis", {}),
                "report": state.get("report", {}),
                "review_feedback": state.get("review_feedback", {}),
                "execution_log": state.get("execution_log", []),
            },
            warnings=state.get("warnings", []),
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
                "rag_ingest",
            ],
            "current_capacity": "LangGraph StateGraph with competitor fan-out + RAG write-back",
            "profile_subgraph": "fan_out -> search_worker | scrape_worker | rag_retriever -> aggregator",
            "rag_loop": "each approved report is ingested back into Qdrant as historical analysis memory",
            "checkpointer": type(self.checkpointer).__name__,
        }

    def export_mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()

    def get_checkpoint(self, query: str) -> Any:
        return self.checkpointer.get_tuple(self._config(query))

    def _build_profile_graph(self) -> Any:
        graph = StateGraph(CompetitorProfileGraphState)
        graph.add_node("search_worker", self._profile_search_node)
        graph.add_node("scrape_worker", self._profile_scrape_node)
        graph.add_node("rag_retriever", self._profile_rag_node)
        graph.add_node("aggregator", self._profile_aggregator_node)
        graph.add_node("fan_out", lambda s: {})
        graph.add_edge(START, "fan_out")
        graph.add_edge("fan_out", "search_worker")
        graph.add_edge("fan_out", "scrape_worker")
        graph.add_edge("fan_out", "rag_retriever")
        graph.add_edge("search_worker", "aggregator")
        graph.add_edge("scrape_worker", "aggregator")
        graph.add_edge("rag_retriever", "aggregator")
        graph.add_edge("aggregator", END)
        return graph.compile()

    def _build_graph(self) -> Any:
        graph = StateGraph(CompIntelState)
        graph.add_node("intent_analyst", self._intent_node)
        graph.add_node("research_planner", self._planner_node)
        graph.add_node("competitor_profiler", self._profile_one_node)
        graph.add_node("market_analyst", self._market_node)
        graph.add_node("swot_synthesizer", self._swot_node)
        graph.add_node("report_writer", self._report_node)
        graph.add_node("reviewer", self._review_node)
        graph.add_node("rag_ingest", self._rag_ingest_node)

        graph.add_edge(START, "intent_analyst")
        graph.add_edge("intent_analyst", "research_planner")
        graph.add_conditional_edges("research_planner", self._fan_out_competitors)
        graph.add_edge("competitor_profiler", "market_analyst")
        graph.add_edge("market_analyst", "swot_synthesizer")
        graph.add_edge("swot_synthesizer", "report_writer")
        graph.add_edge("report_writer", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            self._review_route,
            {"approved": "rag_ingest", "revise": "report_writer"},
        )
        graph.add_edge("rag_ingest", END)
        return graph.compile(checkpointer=self.checkpointer)

    async def _intent_node(self, state: CompIntelState) -> dict[str, Any]:
        result = await self.intent_analyst(state.get("query", ""))
        intent = result.get("intent") or {}
        return {
            "intent": intent,
            "target": result.get("target") or intent.get("target"),
            "market_segment": result.get("market_segment") or intent.get("market_segment"),
            "competitors": result.get("competitors", []),
            "research_questions": result.get("research_questions", []),
            "warnings": result.get("notes", []),
            "execution_log": [
                {"node": "intent_analyst", "event": "completed", "detail": result.get("target", "unknown")}
            ],
        }

    async def _planner_node(self, state: CompIntelState) -> dict[str, Any]:
        return await self.research_planner(state)

    def _fan_out_competitors(self, state: CompIntelState) -> list[Send] | Literal["market_analyst"]:
        competitors = state.get("competitors", [])
        if not competitors:
            return "market_analyst"
        return [
            Send(
                "competitor_profiler",
                {
                    "competitor": competitor,
                    "research_questions": state.get("research_questions", []),
                    "market_segment": state.get("market_segment", ""),
                },
            )
            for competitor in competitors
        ]

    async def _profile_one_node(self, state: CompIntelState) -> dict[str, Any]:
        result = await self.profile_app.ainvoke(
            {
                "competitor": state.get("competitor") or {},
                "research_questions": state.get("research_questions", []),
                "market_segment": state.get("market_segment", ""),
                "execution_log": [],
            }
        )
        profile = result.get("profile", {})
        return {
            "profiles": [profile],
            "execution_log": result.get("execution_log", []),
        }

    async def _profile_search_node(self, state: CompetitorProfileGraphState) -> dict[str, Any]:
        result = await self.competitor_profiler.search_worker(
            {
                "competitor": state.get("competitor") or {},
                "research_questions": state.get("research_questions", []),
            }
        )
        return {
            "search_results": result,
            "execution_log": result.get("execution_log", []),
        }

    async def _profile_scrape_node(self, state: CompetitorProfileGraphState) -> dict[str, Any]:
        result = await self.competitor_profiler.scrape_worker(
            {"competitor": state.get("competitor") or {}}
        )
        return {
            "scraped_content": result,
            "execution_log": result.get("execution_log", []),
        }

    async def _profile_rag_node(self, state: CompetitorProfileGraphState) -> dict[str, Any]:
        result = await self.competitor_profiler.rag_retriever(
            {
                "competitor": state.get("competitor") or {},
                "market_segment": state.get("market_segment", ""),
            }
        )
        return {
            "rag_context": result,
            "execution_log": result.get("execution_log", []),
        }

    async def _profile_aggregator_node(self, state: CompetitorProfileGraphState) -> dict[str, Any]:
        competitor = state.get("competitor") or {}
        search = state.get("search_results") or {}
        scrape = state.get("scraped_content") or {}
        rag = state.get("rag_context") or {}
        name = competitor.get("name", "unknown")
        profile = {
            "name": name,
            "website": competitor.get("website"),
            "summary": f"Profile summary for {name}.",
            "search_results": search.get("search_results", []),
            "scraped_content": scrape.get("scraped_content", []),
            "rag_context": rag.get("rag_context", []),
            "sources": ["search_worker", "scrape_worker", "rag_retriever"],
        }
        return {
            "profile": profile,
            "execution_log": [
                {"node": "competitor_profiler", "event": "completed", "detail": name}
            ],
        }

    async def _market_node(self, state: CompIntelState) -> dict[str, Any]:
        return await self.market_analyst(
            {
                "profiles": state.get("profiles", []),
                "market_segment": state.get("market_segment"),
            }
        )

    async def _swot_node(self, state: CompIntelState) -> dict[str, Any]:
        return await self.swot_synthesizer(
            {
                "profiles": state.get("profiles", []),
                "market_analysis": state.get("market_analysis", {}),
            }
        )

    async def _report_node(self, state: CompIntelState) -> dict[str, Any]:
        return await self.report_writer(
            {
                "query": state.get("query"),
                "intent": state.get("intent", {}),
                "profiles": state.get("profiles", []),
                "market_analysis": state.get("market_analysis", {}),
                "swot_analysis": state.get("swot_analysis", {}),
                "review_feedback": state.get("review_feedback", {}),
            }
        )

    async def _review_node(self, state: CompIntelState) -> dict[str, Any]:
        result = await self.reviewer(
            {
                "report": state.get("report", {}),
                "review_feedback": state.get("review_feedback", {}),
            }
        )
        feedback = result.get("review_feedback", {})
        retry_count = int(feedback.get("retry_count", 0))
        if not feedback.get("approved"):
            feedback["retry_count"] = retry_count + 1
        return {
            "review_feedback": feedback,
            "retry_count": int(feedback.get("retry_count", retry_count)),
            "execution_log": result.get("execution_log", []),
        }

    def _review_route(self, state: CompIntelState) -> Literal["approved", "revise"]:
        feedback = state.get("review_feedback", {})
        if feedback.get("approved") or int(feedback.get("retry_count", 0)) >= ReviewerAgent.MAX_RETRIES:
            return "approved"
        return "revise"

    async def _rag_ingest_node(self, state: CompIntelState) -> dict[str, Any]:
        """Ingest the approved report into Qdrant as historical analysis memory.

        This is the write-path of the RAG loop: every completed analysis
        feeds back into the vector store so that future queries in the
        same market segment can retrieve past insights.
        """
        target = state.get("target") or state.get("intent", {}).get("target", "unknown")
        market_segment = state.get("market_segment", "")
        report = state.get("report", {})
        swot = state.get("swot_analysis", {})
        market = state.get("market_analysis", {})
        profiles = state.get("profiles", [])

        documents = []
        now = datetime.now(timezone.utc).isoformat()

        # Ingest executive summary / conclusion
        exec_summary = report.get("executive_summary", "")
        conclusion = report.get("conclusion", "")
        if exec_summary or conclusion:
            documents.append({
                "text": f"Target: {target}\nMarket: {market_segment}\nExecutive Summary: {exec_summary}\nConclusion: {conclusion}",
                "source": f"report:{target}:{now}",
                "metadata": {"report_type": "executive_summary", "target": target,
                             "market_segment": market_segment, "ingested_at": now},
            })

        # Ingest SWOT per competitor
        for comp in (swot.get("competitors") or []):
            if isinstance(comp, dict) and comp.get("name"):
                documents.append({
                    "text": f"SWOT for {comp.get('name')}: {comp}",
                    "source": f"swot:{comp.get('name')}:{now}",
                    "metadata": {"report_type": "swot", "target": target,
                                 "competitor": comp.get("name"),
                                 "market_segment": market_segment, "ingested_at": now},
                })

        # Ingest market analysis
        if market:
            documents.append({
                "text": f"Market analysis for {market_segment}: {market}",
                "source": f"market:{target}:{now}",
                "metadata": {"report_type": "market_analysis", "target": target,
                             "market_segment": market_segment, "ingested_at": now},
            })

        if documents:
            try:
                ingested = self.competitor_profiler.rag_retriever.store.ingest(documents)
                return {
                    "execution_log": [{
                        "node": "rag_ingest", "event": "completed",
                        "detail": f"Ingested {ingested} chunks into RAG memory for {target}",
                    }]
                }
            except Exception as exc:
                return {
                    "execution_log": [{
                        "node": "rag_ingest", "event": "completed_with_error",
                        "detail": f"RAG ingest failed (non-fatal): {exc}",
                    }]
                }
        return {"execution_log": [{"node": "rag_ingest", "event": "completed", "detail": "no documents to ingest"}]}

    def _config(self, query: str) -> dict[str, Any]:
        thread_id = f"compintel:{abs(hash(query))}"
        return {"configurable": {"thread_id": thread_id}}

