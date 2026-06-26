"""OpenAI-compatible LLM client for CompIntel Research."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
from urllib.request import Request, urlopen

from .settings import CompIntelSettings


async def create_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    llm_provider: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout: float | None = None,
    thinking: dict[str, str] | None = {"type": "disabled"},
    response_format: dict[str, str] | None = None,
    **_: Any,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    thinking:
        DeepSeek V4 thinking control, passed via ``extra_body``.
        ``{"type": "disabled"}`` prevents the model from burning tokens
        on chain-of-thought; ``{"type": "enabled"}`` enables reasoning.
        Defaults to disabled to prevent spontaneous reasoning mode.
    response_format:
        OpenAI-compatible JSON mode.  ``{"type": "json_object"}``
        guarantees the model output is valid JSON (V4 non-thinking only).
    """

    settings = CompIntelSettings.from_env()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not configured.")
    if not settings.llm_base_url:
        raise RuntimeError("LLM_BASE_URL is not configured.")

    endpoint = _chat_endpoint(settings.llm_base_url)
    _, model_name = _split_provider_model(model)
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking:
        # DeepSeek requires thinking to be nested in extra_body,
        # not at the top level of the request.
        payload["extra_body"] = {"thinking": thinking}
    if response_format:
        payload["response_format"] = response_format
    effective_timeout = timeout if timeout is not None else settings.llm_timeout_seconds
    return await asyncio.to_thread(
        _post_chat_completion,
        endpoint,
        settings.llm_api_key,
        payload,
        effective_timeout,
    )


async def create_chat_completion_raw(
    messages: list[dict[str, str]],
    model: str,
    llm_provider: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout: float | None = None,
    thinking: dict[str, str] | None = {"type": "disabled"},
    **_: Any,
) -> dict[str, str | None]:
    """Like :func:`create_chat_completion` but returns the raw message dict
    with both ``content`` and ``reasoning_content`` keys.

    This is used by :meth:`LLMService.call_with_reasoning` to capture
    chain-of-thought from reasoning models so it can be passed to a
    formatting model.
    """
    settings = CompIntelSettings.from_env()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not configured.")
    if not settings.llm_base_url:
        raise RuntimeError("LLM_BASE_URL is not configured.")

    endpoint = _chat_endpoint(settings.llm_base_url)
    _, model_name = _split_provider_model(model)
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking:
        payload["extra_body"] = {"thinking": thinking}
    effective_timeout = timeout if timeout is not None else settings.llm_timeout_seconds
    return await asyncio.to_thread(
        _post_chat_completion_raw,
        endpoint,
        settings.llm_api_key,
        payload,
        effective_timeout,
    )


def _post_chat_completion_raw(
    endpoint: str, api_key: str, payload: dict[str, Any], timeout: float
) -> dict[str, str | None]:
    """Same retry logic as :func:`_post_chat_completion`, but returns the
    raw message dict ``{'content': ..., 'reasoning_content': ...}`` without
    any post-processing.
    """
    import time

    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_network_error: Exception | None = None
    data = None
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        request = Request(endpoint, data=payload_bytes, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM provider returned HTTP {exc.code}: {detail[:500]}") from None
        except (OSError, URLError, IncompleteRead) as exc:
            last_network_error = exc
            if attempt < max_attempts:
                backoff = 0.5 * (2 ** (attempt - 1))
                time.sleep(backoff)
                continue
            raise RuntimeError(
                f'{{"error": "LLM provider unavailable after {max_attempts} attempts", '
                f'"detail": "{exc}", '
                f'"recovery": "Try again in 30 seconds or check your LLM_BASE_URL configuration."}}'
            ) from None
        else:
            break  # success — exit retry loop

    if last_network_error is not None and data is None:  # pragma: no cover — defensive
        raise RuntimeError(
            f'{{"error": "LLM provider unavailable after {max_attempts} attempts", '
            f'"detail": "{last_network_error}", '
            f'"recovery": "Try again in 30 seconds or check your LLM_BASE_URL configuration."}}'
        ) from None

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("LLM provider returned no choices.")
    message = choices[0].get("message") or {}
    return {
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
    }


def _post_chat_completion(endpoint: str, api_key: str, payload: dict[str, Any], timeout: float) -> str:
    """Call the chat completions endpoint with retry, returning the processed
    content string.

    Delegates to :func:`_post_chat_completion_raw` for the HTTP call, then
    applies content / reasoning extraction logic.
    """
    raw = _post_chat_completion_raw(endpoint, api_key, payload, timeout)
    content = raw["content"]
    reasoning = raw["reasoning_content"]

    # V4 models (v4-pro, v4-flash) may spontaneously enter reasoning
    # mode, putting the actual answer in reasoning_content and leaving
    # content empty.  Try to extract JSON from the reasoning tail.
    if not content and reasoning:
        content = _extract_tail_json(reasoning)
        if content is None:
            # Only treat *intentional* reasoning as valid output.
            # Spontaneous reasoning without extractable JSON means the
            # model burned all tokens on chain-of-thought.
            extra_body = payload.get("extra_body", {})
            thinking_cfg = extra_body.get("thinking", {})
            if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "enabled":
                # Reasoner was asked to think — the reasoning IS the output
                content = reasoning
            else:
                raise RuntimeError(
                    "Model spontaneously entered reasoning mode and "
                    "consumed all tokens on chain-of-thought without "
                    "producing structured output.  The caller should "
                    "retry with a shorter prompt or a different model."
                )
    if content is None:
        raise RuntimeError("LLM provider returned no message content.")
    # DeepSeek reasoning models sometimes return a whitespace-only string
    content_str = str(content).strip()
    if not content_str:
        raise RuntimeError(
            "LLM provider returned empty content. The reasoning model may have "
            "consumed all tokens on chain-of-thought without producing a final "
            "answer. Consider reducing max_tokens for the reasoning phase or "
            "switching to a non-reasoning model for this task."
        )
    return content_str


def _extract_tail_json(text: str) -> str | None:
    """Extract the last JSON object ``{...}`` from *text*.

    Reasoning models (DeepSeek-R1, o1, etc.) emit chain-of-thought followed
    by the actual answer.  This helper finds the final complete JSON object
    in the response so we can discard the reasoning preamble.
    """
    # Find all { } pairs and pick the last one that looks like a
    # complete JSON object (brace count returns to 0).
    best_start = -1
    best_end = -1
    brace_depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if brace_depth == 0:
                best_start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and best_start >= 0:
                best_end = i + 1
    if best_start >= 0 and best_end > best_start:
        candidate = text[best_start:best_end].strip()
        # Quick sanity: at least 8 chars (shortest valid JSON: {"a":1})
        if len(candidate) >= 8:
            return candidate
    return None


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
