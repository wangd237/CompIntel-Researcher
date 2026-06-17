"""RAG primitives for CompIntel Research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class RagDocument:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Embedder(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError
