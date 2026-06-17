"""OpenAI-compatible LLM client for CompIntel Research."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .settings import CompIntelSettings


async def create_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    llm_provider: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    **_: Any,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    The provider value is currently informational because DeepSeek, Kimi, GLM,
    and custom compatible endpoints all expose the same request shape here.
    """

    settings = CompIntelSettings.from_env()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not configured.")
    if not settings.llm_base_url:
        raise RuntimeError("LLM_BASE_URL is not configured.")

    endpoint = _chat_endpoint(settings.llm_base_url)
    _, model_name = _split_provider_model(model)
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return await asyncio.to_thread(
        _post_chat_completion,
        endpoint,
        settings.llm_api_key,
        payload,
        settings.llm_timeout_seconds,
    )


def _post_chat_completion(endpoint: str, api_key: str, payload: dict[str, Any], timeout: float) -> str:
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM provider returned HTTP {exc.code}: {detail[:500]}") from None
    except (OSError, URLError) as exc:
        raise RuntimeError(f"LLM provider request failed: {exc}") from None

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("LLM provider returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        content = choices[0].get("text")
    if content is None:
        raise RuntimeError("LLM provider returned no message content.")
    return str(content)


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _split_provider_model(value: str) -> tuple[str, str]:
    if ":" in value:
        provider, model = value.split(":", 1)
        return provider.strip() or "openai", model.strip() or "gpt-4o-mini"
    return "openai", value.strip() or "gpt-4o-mini"
