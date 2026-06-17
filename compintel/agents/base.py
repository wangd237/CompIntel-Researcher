"""Shared agent abstractions for CompIntel Research."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCompIntelAgent(ABC):
    """Common surface for CompIntel agents."""

    def __init__(self, model: str = "deepseek-chat") -> None:
        self.model = model

    @abstractmethod
    async def __call__(self, state: Any) -> dict[str, Any]:
        raise NotImplementedError
