"""CompIntel Core — shared infrastructure for all agents.

Provides the unified :class:`LLMService` (single entry point for LLM calls,
retry, JSON parsing, and three-tier degradation) and :class:`StateAdapter`
(typed wrapper around the raw graph state dict).
"""

from __future__ import annotations

from compintel.core.llm_service import LLMService
from compintel.core.state_adapter import StateAdapter

__all__ = ["LLMService", "StateAdapter"]
