"""RAG retrieval for competitor profiling — searches historical analysis memory."""

from __future__ import annotations

from typing import Any

from ..rag import QdrantStore
from .base import BaseCompIntelAgent


class RAGRetriever(BaseCompIntelAgent):
    """Retrieve relevant historical analyses from the Qdrant vector store.

    The query is built from the competitor name and market segment so that
    past reports in the same industry are recalled — not just exact name
    matches.  Seed data loading is the caller's responsibility; this class
    searches whatever is already in the store.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        store: QdrantStore | None = None,
        top_k: int = 5,
    ) -> None:
        super().__init__(model=model)
        self.store = store or QdrantStore()
        self.top_k = top_k

    async def __call__(self, state: Any) -> dict[str, Any]:
        competitor = {}
        market_segment = ""
        if isinstance(state, dict):
            competitor = state.get("competitor") or {}
            market_segment = str(state.get("market_segment", "")).strip()

        name = competitor.get("name", "unknown")
        query = f"{market_segment} {name}".strip() if market_segment else str(name)

        try:
            rag_context = self.store.similarity_search(query, top_k=self.top_k)
        except Exception as exc:
            return {
                "rag_context": [],
                "execution_log": [
                    {
                        "node": "rag_retriever",
                        "event": "completed_with_error",
                        "detail": str(exc),
                    }
                ],
            }

        return {
            "rag_context": rag_context,
            "execution_log": [
                {
                    "node": "rag_retriever",
                    "event": "completed",
                    "detail": f"{len(rag_context)} contexts retrieved for {name}",
                }
            ],
        }
