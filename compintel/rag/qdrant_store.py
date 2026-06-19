"""Qdrant-backed vector store for CompIntel RAG."""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .base import Embedder, RagDocument

logger = logging.getLogger(__name__)


class HashEmbedder:
    """Deterministic lightweight embedder used when no model adapter is wired."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token for token in text.lower().split() if token]
        for token in tokens or [text.lower() or "empty"]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    """Semantic embedder backed by a local sentence-transformers model.

    The model is lazy-loaded on first ``embed()`` call so that imports and
    tests stay fast when no RAG indexing is happening.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    @property
    def dimensions(self) -> int:
        self._ensure_model()
        model = self._model
        # sentence-transformers renamed this method across versions
        dim = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension", None)
        return dim() if dim else 0

    def embed(self, text: str) -> list[float]:
        self._ensure_model()
        return self._model.encode(text, normalize_embeddings=True).tolist()  # type: ignore[union-attr]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from None
        self._model = SentenceTransformer(self.model_name, device=self.device)


def _resolve_embedder(explicit: Embedder | None = None) -> Embedder:
    if explicit is not None:
        return explicit

    from ..settings import CompIntelSettings  # deferred to keep rag/ decoupled from settings
    settings = CompIntelSettings.from_env()
    model_name = settings.embedding_model.strip()
    if model_name:
        logger.info("Loading embedding model: %s", model_name)
        return SentenceTransformerEmbedder(model_name=model_name)

    return HashEmbedder()


@dataclass(slots=True)
class QdrantStore:
    collection_name: str = "compintel_reports"
    client: QdrantClient | None = None
    embedder: Embedder = field(default_factory=_resolve_embedder)
    location: str = ":memory:"
    chunk_size: int = 900
    chunk_overlap: int = 120

    def __post_init__(self) -> None:
        if self.client is None:
            if self.location == ":memory:":
                self.client = QdrantClient(location=":memory:")
            else:
                self.client = QdrantClient(path=self.location)

    @classmethod
    def from_settings(cls, collection_name: str = "compintel_reports") -> "QdrantStore":
        """Create a store whose location is read from CompIntelSettings.

        Falls back to ``:memory:`` when no explicit QDRANT_PATH is configured,
        so that tests and ad-hoc runs stay lightweight.
        """
        from ..settings import CompIntelSettings  # deferred import
        settings = CompIntelSettings.from_env()
        path = (settings.qdrant_path or "").strip()
        if path:
            return cls(collection_name=collection_name, location=path)
        return cls(collection_name=collection_name)

    def ensure_collection(self) -> None:
        assert self.client is not None
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedder.dimensions,
                distance=Distance.COSINE,
            ),
        )

    def ingest(self, documents: Iterable[RagDocument | dict[str, Any]], batch_size: int = 100) -> int:
        self.ensure_collection()
        points: list[PointStruct] = []
        total = 0
        for document in documents:
            rag_document = self._coerce_document(document)
            for chunk_index, chunk in enumerate(self._chunk_text(rag_document.text)):
                point_id = self._point_id(rag_document.source, chunk_index, chunk)
                payload = {
                    "text": chunk,
                    "source": rag_document.source,
                    "chunk_index": chunk_index,
                    **rag_document.metadata,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=self.embedder.embed(chunk),
                        payload=payload,
                    )
                )
                if len(points) >= batch_size:
                    self._upsert(points)
                    total += len(points)
                    points = []

        if points:
            self._upsert(points)
            total += len(points)
        return total

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.ensure_collection()
        assert self.client is not None
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedder.embed(query),
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": point.payload.get("text", "") if point.payload else "",
                "source": point.payload.get("source", "") if point.payload else "",
                "score": float(point.score),
                "metadata": {
                    key: value
                    for key, value in (point.payload or {}).items()
                    if key not in {"text", "source"}
                },
            }
            for point in response.points
        ]

    def _upsert(self, points: list[PointStruct]) -> None:
        assert self.client is not None
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def _coerce_document(self, document: RagDocument | dict[str, Any]) -> RagDocument:
        if isinstance(document, RagDocument):
            return document
        return RagDocument(
            text=str(document.get("text", "")),
            source=str(document.get("source", "unknown")),
            metadata=dict(document.get("metadata", {})),
        )

    def _chunk_text(self, text: str) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(normalized):
            chunk = normalized[start : start + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    def _point_id(self, source: str, chunk_index: int, chunk: str) -> int:
        raw = f"{source}:{chunk_index}:{chunk}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
