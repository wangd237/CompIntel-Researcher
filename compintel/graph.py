"""LangGraph workflow assembly for CompIntel Research."""

from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

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


class CompetitorProfileGraphState(TypedDict):
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
    curator: CuratorAgent = field(init=False, repr=False)
    editor: EditorAgent = field(init=False, repr=False)
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
                GraphNode("curator", "Clean profiles and grade evidence quality"),
                GraphNode("market_analyst", "Aggregate market landscape"),
                GraphNode("swot_synthesizer", "Build SWOT matrix (per competitor)"),
                GraphNode("report_writer", "Write the final report"),
                GraphNode("editor", "Editorial pass: unify voice, resolve conflicts"),
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
        self.curator = CuratorAgent(model=self.model)
        self.editor = EditorAgent(model=self.model)
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

    @staticmethod
    def _detect_language(query: str) -> str:
        """Return 'zh' if the query is predominantly Chinese, 'en' otherwise."""
        chinese_chars = sum(1 for c in query if '一' <= c <= '鿯')
        return "zh" if chinese_chars >= 2 else "en"

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
                "curator_evidence": state.get("curator_evidence", {}),
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
                "curator",
                "market_analyst",
                "swot_synthesizer",
                "report_writer",
                "editor",
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
        # New: competitor_profiler → curator (clean & grade) → market_analyst
        graph.add_edge("competitor_profiler", "curator")
        graph.add_edge("curator", "market_analyst")
        graph.add_edge("market_analyst", "swot_synthesizer")
        graph.add_edge("swot_synthesizer", "report_writer")
        # New: report_writer → editor (unify voice, resolve conflicts) → reviewer
        graph.add_edge("report_writer", "editor")
        graph.add_edge("editor", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            self._review_route,
            {"approved": "rag_ingest", "revise": "report_writer"},
        )
        graph.add_edge("rag_ingest", END)
        return graph.compile(checkpointer=self.checkpointer)

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
        try:
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
        except Exception as exc:
            competitor = state.get("competitor") or {}
            name = competitor.get("name", "unknown")
            profile = {
                "name": name,
                "summary": f"{name} profiling skipped — source unavailable",
            }
            return {
                "profiles": [profile],
                "warnings": [f"{name} profiling skipped — source unavailable. Re-run with different search terms to include this competitor."],
                "execution_log": [
                    {"node": "competitor_profiler", "event": "error", "detail": f"profiling failed: {exc}"}
                ],
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
            {
                "competitor": state.get("competitor") or {},
                "market_segment": state.get("market_segment", ""),
            }
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
        summary = self._build_profile_summary(name, search, scrape, rag)
        profile = {
            "name": name,
            "website": competitor.get("website"),
            "summary": summary,
            "search_results": search.get("search_results", []),
            "scraped_content": scrape.get("scraped_content", []),
            "rag_context": rag.get("rag_context", []),
            "sources": self._extract_profile_sources(search, scrape, rag),
        }
        return {
            "profile": profile,
            "execution_log": [
                {"node": "competitor_profiler", "event": "completed", "detail": name}
            ],
        }

    @staticmethod
    def _extract_profile_sources(
        search: dict[str, Any],
        scrape: dict[str, Any],
        rag: dict[str, Any],
    ) -> list[str]:
        """Collect real URLs from all three data channels."""
        sources: list[str] = []
        seen: set[str] = set()
        for item in (search.get("search_results") or [])[:5]:
            url = str(item.get("href") or item.get("url", "")).strip()
            if url and url not in seen and url.startswith("http"):
                sources.append(url)
                seen.add(url)
        for item in (scrape.get("scraped_content") or [])[:3]:
            url = str(item.get("url") or item.get("source", "")).strip()
            if url and url not in seen and url.startswith("http"):
                sources.append(url)
                seen.add(url)
        for item in (rag.get("rag_context") or [])[:3]:
            url = str(item.get("source", "")).strip()
            if url and url not in seen and url.startswith("http"):
                sources.append(url)
                seen.add(url)
        return sources if sources else ["search_worker", "scrape_worker", "rag_retriever"]

    @staticmethod
    def _build_profile_summary(
        name: str,
        search: dict[str, Any],
        scrape: dict[str, Any],
        rag: dict[str, Any],
    ) -> str:
        """Build a data-derived profile summary from collected sources.

        Extracts the most informative snippets from search results,
        scraped content, and RAG context — no LLM call, deterministic.
        """
        parts: list[str] = [f"{name}"]

        # 1. Best search snippet (title + body)
        search_results = search.get("search_results", []) or []
        for item in search_results[:3]:
            if isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                snippet = str(item.get("body") or item.get("snippet", "")).strip()
                if title:
                    parts.append(title)
                if snippet:
                    parts.append(snippet[:200])

        # 2. Key scraped text (first meaningful chunk)
        scraped = scrape.get("scraped_content", []) or []
        for item in scraped[:2]:
            if isinstance(item, dict):
                content = str(item.get("raw_content") or item.get("content", "")).strip()
                if content and len(content) > 20:
                    # Take the first ~300 chars that aren't just nav boilerplate
                    lines = [ln.strip() for ln in content.split("\n") if len(ln.strip()) > 15]
                    if lines:
                        parts.append("\n".join(lines[:4]))

        # 3. RAG context (best match — skip polluted old report metadata)
        rag_context = rag.get("rag_context", []) or []
        for item in rag_context[:2]:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text and len(text) > 10:
                    # Skip entries that are just old report metadata dumps
                    # or cross-domain pollution (e.g. GPU market analysis for an e-commerce query).
                    if "Executive Summary:" in text or "Target: " in text:
                        continue
                    parts.append(text[:300])

        # Join with separators — if only name was collected, note the data gap
        if len(parts) == 1:
            return f"{name} — insufficient data collected from search, scrape, or RAG."
        return " | ".join(parts)

    async def _curator_node(self, state: CompIntelState) -> dict[str, Any]:
        """Clean profiles and grade evidence quality after fan-out."""
        return await self.curator(
            {
                "profiles": state.get("profiles", []),
                "market_segment": state.get("market_segment"),
                "language": state.get("language", "en"),
            }
        )

    async def _market_node(self, state: CompIntelState) -> dict[str, Any]:
        profiles = state.get("curated_profiles") or state.get("profiles", [])
        return await self.market_analyst(
            {
                "profiles": profiles,
                "market_segment": state.get("market_segment"),
                "language": state.get("language", "en"),
            }
        )

    async def _swot_node(self, state: CompIntelState) -> dict[str, Any]:
        profiles = state.get("curated_profiles") or state.get("profiles", [])
        return await self.swot_synthesizer(
            {
                "profiles": profiles,
                "market_analysis": state.get("market_analysis", {}),
                "language": state.get("language", "en"),
            }
        )

    async def _report_node(self, state: CompIntelState) -> dict[str, Any]:
        profiles = state.get("curated_profiles") or state.get("profiles", [])
        return await self.report_writer(
            {
                "query": state.get("query"),
                "intent": state.get("intent", {}),
                "profiles": profiles,
                "market_analysis": state.get("market_analysis", {}),
                "swot_analysis": state.get("swot_analysis", {}),
                "review_feedback": state.get("review_feedback", {}),
                "language": state.get("language", "en"),
            }
        )

    async def _editor_node(self, state: CompIntelState) -> dict[str, Any]:
        """Editorial pass: unify voice, remove duplication, resolve conflicts."""
        return await self.editor(
            {
                "report": state.get("report", {}),
                "language": state.get("language", "en"),
            }
        )

    async def _review_node(self, state: CompIntelState) -> dict[str, Any]:
        result = await self.reviewer(
            {
                "report": state.get("report", {}),
                "review_feedback": state.get("review_feedback", {}),
                "retry_count": state.get("retry_count", 0),
            }
        )
        feedback = result.get("review_feedback", {})
        retry_count = int(state.get("retry_count", 0))
        if not feedback.get("approved"):
            retry_count += 1
        # P1-3: Auto-approval banner when retry limit is exhausted
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

        # Guard: only ingest reports that passed reviewer approval.
        # Writing unapproved reports pollutes the RAG index for future queries.
        # P2-2: Allow auto-approved reports (retry_count exhausted) but tag them as limited quality.
        feedback = state.get("review_feedback", {})
        retry_count = int(state.get("retry_count", 0))
        is_auto_approved = retry_count >= ReviewerAgent.MAX_RETRIES and not feedback.get("approved")
        if not feedback.get("approved") and not is_auto_approved:
            return {
                "execution_log": [{
                    "node": "rag_ingest", "event": "completed",
                    "detail": f"Skipped ingest — report was not approved by reviewer",
                }]
            }

        quality = "limited" if is_auto_approved else "reviewed"
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
                             "market_segment": market_segment, "ingested_at": now,
                             "quality": quality},
            })

        # Ingest SWOT per competitor
        for comp in (swot.get("competitors") or []):
            if isinstance(comp, dict) and comp.get("name"):
                documents.append({
                    "text": f"SWOT for {comp.get('name')}: {comp}",
                    "source": f"swot:{comp.get('name')}:{now}",
                    "metadata": {"report_type": "swot", "target": target,
                                 "competitor": comp.get("name"),
                                 "market_segment": market_segment, "ingested_at": now,
                                 "quality": quality},
                })

        # Ingest market analysis
        if market:
            documents.append({
                "text": f"Market analysis for {market_segment}: {market}",
                "source": f"market:{target}:{now}",
                "metadata": {"report_type": "market_analysis", "target": target,
                             "market_segment": market_segment, "ingested_at": now,
                             "quality": quality},
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
        thread_id = f"compintel:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        return {"configurable": {"thread_id": thread_id}}

