"""Environment-backed settings for CompIntel Research."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class CompIntelSettings:
    fast_llm: str = "openai:gpt-4o-mini"
    smart_llm: str = "openai:gpt-4.1"
    strategic_llm: str = "openai:o4-mini"
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    report_store_path: str = "outputs/compintel_audit.jsonl"

    @classmethod
    def from_env(cls) -> "CompIntelSettings":
        return cls(
            fast_llm=os.getenv("FAST_LLM", "openai:gpt-4o-mini"),
            smart_llm=os.getenv("SMART_LLM", "openai:gpt-4.1"),
            strategic_llm=os.getenv("STRATEGIC_LLM", "openai:o4-mini"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            report_store_path=os.getenv("COMPINTEL_AUDIT_PATH", "outputs/compintel_audit.jsonl"),
        )
