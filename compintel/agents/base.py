"""Shared agent abstractions for CompIntel Research.

Provides :class:`BaseCompIntelAgent` — the common surface for all
CompIntel agents with lazy :class:`LLMService` injection and typed
:class:`StateAdapter` extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from compintel.core.llm_service import LLMService
    from compintel.core.state_adapter import StateAdapter


class BaseCompIntelAgent(ABC):
    """Common surface for CompIntel agents.

    Every agent inherits from this class and implements ``__call__``.

    Parameters
    ----------
    model:
        Raw model string (backward-compatible with existing agent
        constructors).  Deprecated in favour of *model_key* + LLMService.
    model_key:
        Symbolic model key — ``"fast"``, ``"smart"``, or ``"strategic"``.
        Used by :attr:`llm` when the agent delegates to ``LLMService``.
    llm_service:
        Pre-configured :class:`LLMService` instance.  When *None* the
        agent lazily creates one from ``CompIntelSettings.from_env()`` on
        first access to :attr:`llm`.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        model_key: str = "smart",
        llm_service: LLMService | None = None,
    ) -> None:
        self.model = model  # backward-compat; kept for existing sub-class constructors
        self.model_key = model_key
        self._llm = llm_service

    @property
    def llm(self) -> LLMService:
        """Lazy :class:`LLMService` — created on first access from env settings."""
        if self._llm is None:
            from compintel.core.llm_service import LLMService  # local import avoids circularity
            from compintel.settings import CompIntelSettings

            self._llm = LLMService(CompIntelSettings.from_env())
        return self._llm

    def read_state(self, state: Any) -> StateAdapter:
        """Wrap *state* in a typed :class:`StateAdapter`.

        Eliminates ``isinstance(state, dict)`` guards and ``.get()``
        chains inside agent ``__call__`` implementations.
        """
        from compintel.core.state_adapter import StateAdapter

        return StateAdapter(state if isinstance(state, dict) else {})

    @abstractmethod
    async def __call__(self, state: Any) -> dict[str, Any]:
        raise NotImplementedError
