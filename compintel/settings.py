"""Environment-backed settings for CompIntel Research."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LLM_PROVIDER_PREFIXES = {
    "deepseek": "openai",
    "kimi": "openai",
    "glm": "openai",
    "openai-compatible": "openai",
}


@dataclass(slots=True)
class CompIntelSettings:
    llm_provider: str = "deepseek"
    fast_llm: str = "openai:deepseek-chat"
    smart_llm: str = "openai:deepseek-chat"
    strategic_llm: str = "openai:deepseek-reasoner"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    search_provider: str = "tavily"
    search_api_key: str | None = None
    report_store_path: str = "outputs/compintel_audit.jsonl"

    @property
    def openai_base_url(self) -> str | None:
        return self.llm_base_url

    @property
    def openai_api_key(self) -> str | None:
        return self.llm_api_key

    @classmethod
    def from_env(cls) -> "CompIntelSettings":
        values = _read_dotenv(Path(".env"))
        llm_provider = _setting(values, "LLM_PROVIDER", "deepseek").lower()
        search_provider = _setting(values, "SEARCH_PROVIDER", "tavily").lower()

        return cls(
            llm_provider=llm_provider,
            fast_llm=_normalize_model(_setting(values, "FAST_LLM", "deepseek-chat"), llm_provider),
            smart_llm=_normalize_model(_setting(values, "SMART_LLM", "deepseek-chat"), llm_provider),
            strategic_llm=_normalize_model(_setting(values, "STRATEGIC_LLM", "deepseek-reasoner"), llm_provider),
            llm_base_url=_setting(values, "LLM_BASE_URL", "") or _setting(values, "OPENAI_BASE_URL", "") or None,
            llm_api_key=_clean_secret(
                _setting(values, "LLM_API_KEY", "")
                or _setting(values, "OPENAI_API_KEY", "")
            ),
            search_provider=search_provider,
            search_api_key=_clean_secret(
                _setting(values, "SERPAPI_API_KEY", "")
                or _setting(values, "TAVILY_API_KEY", "")
            ),
            report_store_path=_setting(values, "COMPINTEL_AUDIT_PATH", "outputs/compintel_audit.jsonl"),
        )


def _setting(dotenv: dict[str, str], key: str, default: str) -> str:
    return os.getenv(key) or dotenv.get(key) or default


def _normalize_model(model: str, provider: str) -> str:
    model = model.strip()
    if ":" in model:
        return model
    llm_provider = LLM_PROVIDER_PREFIXES.get(provider, provider)
    return f"{llm_provider}:{model}"


def _clean_secret(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.strip().lower()
    if not lowered or "replace-with" in lowered or "your_" in lowered or "your-" in lowered:
        return None
    return value.strip()


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values
