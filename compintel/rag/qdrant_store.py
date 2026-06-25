"""Qdrant-backed vector store for CompIntel RAG."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

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
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)


_CJK_RE = re.compile(r"[一-鿿]")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BM25SparseEmbedder:
    """Lightweight BM25 sparse embedder with zero external dependencies.

    Produces sparse vectors in the ``(indices, values)`` format that Qdrant
    accepts for hybrid dense + sparse search.  The vocabulary is built
    incrementally during :meth:`encode_document` calls, and document
    frequency statistics are updated so that :meth:`encode_query` can apply
    IDF weighting.

    Parameters
    ----------
    k1: float
        BM25 term-frequency saturation parameter (default 1.5).
    b: float
        BM25 length-normalisation parameter (default 0.75).

    Notes
    -----
    IDF values for previously ingested documents become slightly stale as
    new documents are added, but the sparse path's primary job is catching
    **exact term matches** (precise company names, rare keywords).  A
    marginally stale IDF does not prevent a match from producing a non-zero
    similarity score, and RRF fusion handles score calibration.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}   # token → index
        self._df: dict[int, int] = {}      # index → document frequency
        self._N: int = 0                    # total documents seen
        self._total_length: int = 0         # sum of all document lengths

    # ── public API ─────────────────────────────────────────────────

    def encode_document(self, text: str) -> SparseVector:
        """Encode a single document chunk as a sparse TF vector.

        This method *also* updates the vocabulary and document-frequency
        statistics, so it must be called for every chunk that is ingested.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Update corpus statistics
        self._N += 1
        self._total_length += len(tokens)
        for token in set(tokens):
            idx = self._get_index(token)
            self._df[idx] = self._df.get(idx, 0) + 1

        # Compute log-scaled term frequencies
        tf: dict[int, float] = {}
        for token in tokens:
            idx = self._vocab[token]
            tf[idx] = tf.get(idx, 0.0) + 1.0

        indices: list[int] = []
        values: list[float] = []
        for idx, freq in tf.items():
            indices.append(idx)
            values.append(1.0 + math.log(freq))  # sublinear tf

        return SparseVector(indices=indices, values=values)

    def encode_query(self, query: str) -> SparseVector:
        """Encode a query with IDF-weighted term presence."""
        tokens = self._tokenize(query)
        if not tokens:
            return SparseVector(indices=[], values=[])

        avgdl = self._total_length / max(1, self._N)
        indices: list[int] = []
        values: list[float] = []

        seen: set[int] = set()
        for token in tokens:
            idx = self._vocab.get(token)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            df = self._df.get(idx, 1)
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
            indices.append(idx)
            values.append(idf)

        return SparseVector(indices=indices, values=values)

    # ── helpers ────────────────────────────────────────────────────

    def _get_index(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into a mix of CJK bigrams and ASCII word tokens."""
        text = text.lower().strip()
        if not text:
            return []

        tokens: list[str] = []
        # Extract CJK characters and ASCII word runs
        parts: list[tuple[bool, str]] = []  # (is_cjk, text)
        for ch in text:
            is_cjk = bool(_CJK_RE.match(ch))
            if parts and parts[-1][0] == is_cjk:
                parts[-1] = (is_cjk, parts[-1][1] + ch)
            else:
                parts.append((is_cjk, ch))

        for is_cjk, segment in parts:
            if is_cjk:
                # Bigram for CJK (單字 as fallback when only 1 char)
                for i in range(len(segment)):
                    tokens.append(segment[i])  # unigram
                for i in range(len(segment) - 1):
                    tokens.append(segment[i : i + 2])  # bigram
            else:
                for token in _ASCII_TOKEN_RE.findall(segment):
                    if len(token) >= 2:
                        tokens.append(token)

        return tokens


def _resolve_embedder(explicit: Embedder | None = None) -> Embedder:
    if explicit is not None:
        return explicit

    from ..settings import CompIntelSettings  # deferred to keep rag/ decoupled from settings
    settings = CompIntelSettings.from_env()
    model_name = settings.embedding_model.strip()
    if model_name:
        embedder = SentenceTransformerEmbedder(model_name=model_name)
        logger.info("Using embedding model: %s", model_name)
        return embedder

    return HashEmbedder()


@dataclass(slots=True)
class QdrantStore:
    collection_name: str = "compintel_reports"
    client: QdrantClient | None = None
    embedder: Embedder = field(default_factory=_resolve_embedder)
    sparse_embedder: BM25SparseEmbedder = field(default_factory=BM25SparseEmbedder)
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
        or when the disk path cannot be locked (e.g. during parallel tests).
        """
        from ..settings import CompIntelSettings  # deferred import
        settings = CompIntelSettings.from_env()
        path = (settings.qdrant_path or "").strip()
        if path:
            try:
                return cls(collection_name=collection_name, location=path)
            except Exception:
                logger.warning("Qdrant disk path %s locked, falling back to :memory:", path)
        return cls(collection_name=collection_name)

    def ensure_collection(self) -> None:
        assert self.client is not None
        if self.client.collection_exists(self.collection_name):
            self._validate_or_recreate_collection()
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedder.dimensions,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    def _validate_or_recreate_collection(self) -> None:
        """Ensure the existing collection has named vectors (dense + sparse).

        Collections created before the hybrid-search migration (June 2026)
        used an unnamed default vector.  Ingesting with named vectors on
        such a collection would fail silently.  We detect the mismatch and
        recreate the collection — losing existing data, but preventing a
        cascade of mysterious 500s on every subsequent operation.

        For production persistence (QDRANT_PATH) the user sees a warning;
        for :memory: (tests / dev) the data loss is bounded to the session.
        """
        assert self.client is not None
        try:
            info = self.client.get_collection(self.collection_name)
        except Exception:
            return  # can't introspect — proceed optimistically

        config = info.config
        params = config.params
        if params.vectors is not None and hasattr(params.vectors, "size"):
            # Old-style single unnamed vector — no named "dense" or "sparse"
            logger.warning(
                "Collection %r uses the old unnamed-vector schema.  "
                "Deleting and recreating with named dense+sparse vectors.  "
                "Existing RAG data will be lost — re-run with seed data.",
                self.collection_name,
            )
            self.client.delete_collection(self.collection_name)
            self.ensure_collection()
            return

        # Named-vector collection — verify expected names exist
        vector_names = set(getattr(params.vectors, "keys", lambda: [])())
        sparse_names = {}
        if params.sparse_vectors is not None:
            sparse_names = set(getattr(params.sparse_vectors, "keys", lambda: [])())
        if "dense" not in vector_names or "sparse" not in sparse_names:
            logger.warning(
                "Collection %r is missing expected vector names (has %s, %s).  "
                "Recreating with correct schema.",
                self.collection_name, vector_names, sparse_names,
            )
            self.client.delete_collection(self.collection_name)
            self.ensure_collection()

    def ingest(self, documents: Iterable[RagDocument | dict[str, Any]], batch_size: int = 100) -> int:
        self.ensure_collection()
        points: list[PointStruct] = []
        total = 0
        for document in documents:
            rag_document = self._coerce_document(document)
            for chunk_index, chunk in enumerate(self._chunk_text(rag_document.text)):
                point_id = self._point_id(rag_document.source, chunk_index, chunk)
                payload = {
                    **rag_document.metadata,
                    "text": chunk,
                    "source": rag_document.source,
                    "chunk_index": chunk_index,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": self.embedder.embed(chunk),
                            "sparse": self.sparse_embedder.encode_document(chunk),
                        },
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

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_market_segment: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search with BM25 sparse + Dense semantic vectors fused via RRF.

        Qdrant runs both retrieval paths in parallel, then merges the two
        result lists with Reciprocal Rank Fusion so that documents matching
        both the exact keywords (BM25) and the semantic intent (Dense) are
        ranked highest.
        """
        self.ensure_collection()
        assert self.client is not None

        query_filter: Filter | None = None
        if filter_market_segment:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="market_segment",
                        match=MatchValue(value=filter_market_segment),
                    )
                ]
            )

        dense_vector = self.embedder.embed(query)
        sparse_vector = self.sparse_embedder.encode_query(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=max(top_k * 4, 20),
                    filter=query_filter,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=max(top_k * 4, 20),
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
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
