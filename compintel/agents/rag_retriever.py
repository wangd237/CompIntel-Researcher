"""RAG retrieval for competitor profiling."""

from __future__ import annotations

from typing import Any

from ..rag import QdrantStore
from .base import BaseCompIntelAgent


class RAGRetriever(BaseCompIntelAgent):
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
        if isinstance(state, dict):
            competitor = state.get("competitor") or {}

        name = competitor.get("name", "unknown")
        try:
            rag_context = self.store.similarity_search(str(name), top_k=self.top_k)
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
