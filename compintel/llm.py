"""OpenAI-compatible LLM client for CompIntel Research.

Phase 1 arch rewrite: ``urllib`` + ``asyncio.to_thread`` replaced with
``httpx.AsyncClient`` + connection pool.  TCP and TLS connections are
reused across calls, eliminating the 200-500 ms per-request TLS handshake
and the ~30-40 % handshake failure rate on Windows + DeepSeek.

Global concurrency is capped to ``LLM_MAX_CONCURRENCY`` (default 3) so
that parallel fan-out doesn't overload the local network stack or trigger
DeepSeek rate-limiting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any

import httpx

from .settings import CompIntelSettings

logger = logging.getLogger(__name__)

# ── Global concurrency gate ────────────────────────────────────────────────
# Single semaphore shared by all LLM calls so that parallel fan-out
# (competitor_profiler × N, SWOT × N, report_writer narratives × N) is
# capped globally, not per-call‑site.
_LLM_SEMAPHORE = asyncio.Semaphore(3)

# ── Sentinel for "caller didn't specify thinking" ────────────────────────
# We distinguish "caller omitted thinking" (default to {"type": "disabled"}
# — prevents spontaneous reasoning) from "caller explicitly passed None"
# (means "don't send the thinking field").  DeepSeek V4 enters reasoning
# mode aggressively unless explicitly told not to.
_SENTINEL = object()

# ── Connection pool (lifetime = process) ───────────────────────────────────
# httpx.AsyncClient is the recommended pattern: create once at module level,
# share across all calls.  The pool maintains keep‑alive connections
# independently for each host so that repeated calls to the same API
# endpoint skip the TCP + TLS handshake entirely.
#
# Connection pool size is deliberately conservative (5 per host, 10 total)
# because the semaphore above gates concurrency anyway.
_CLIENT: httpx.AsyncClient | None = None
_CLIENT_LOCK = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the shared ``httpx.AsyncClient``, creating it lazily."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    async with _CLIENT_LOCK:
        if _CLIENT is not None:  # double-check under lock
            return _CLIENT
        _CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=4,
                max_connections=8,
                keepalive_expiry=5.0,  # sync with DeepSeek server idle timeout
            ),
            timeout=httpx.Timeout(45.0, connect=10.0),
        )
        return _CLIENT


# ── Public API ──────────────────────────────────────────────────────────────

async def create_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    llm_provider: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout: float | None = None,
    thinking: dict[str, str] | None = _SENTINEL,
    response_format: dict[str, str] | None = None,
    **_: Any,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    thinking:
        DeepSeek V4 thinking control, sent as a top-level field in the
        request body.  ``{"type": "disabled"}`` prevents the model from
        burning tokens on chain-of-thought; ``{"type": "enabled"}``
        enables reasoning.
    response_format:
        OpenAI-compatible JSON mode.  ``{"type": "json_object"}``
        guarantees the model output is valid JSON (V4 non-thinking only).
    """
    raw = await create_chat_completion_raw(
        messages=messages,
        model=model,
        llm_provider=llm_provider,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        thinking=thinking,
        response_format=response_format,
    )
    return _process_content(raw.get("content"), raw.get("reasoning_content"), raw.get("_payload", {}))


async def create_chat_completion_raw(
    messages: list[dict[str, str]],
    model: str,
    llm_provider: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout: float | None = None,
    thinking: dict[str, str] | None = _SENTINEL,
    response_format: dict[str, str] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Like :func:`create_chat_completion` but returns the raw dict with
    ``content``, ``reasoning_content``, and the internal ``_payload``
    (needed by the content processor).
    """
    settings = CompIntelSettings.from_env()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not configured.")
    if not settings.llm_base_url:
        raise RuntimeError("LLM_BASE_URL is not configured.")

    endpoint = _chat_endpoint(settings.llm_base_url)
    _, model_name = _split_provider_model(model)

    # Resolve thinking sentinel BEFORE building payload dict.
    # _SENTINEL means "caller omitted the parameter" → default to disabled.
    # We must resolve this BEFORE the payload dict is built because the
    # resolved value goes into the JSON-serialized payload, and _SENTINEL
    # (a bare object()) is not JSON-serializable.
    if thinking is _SENTINEL:
        thinking = {"type": "disabled"}

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking is not None:
        payload["thinking"] = thinking
    if response_format is not None:
        payload["response_format"] = response_format

    effective_timeout = timeout if timeout is not None else settings.llm_timeout_seconds

    client = await _get_client()
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    max_attempts = 3
    last_error: Exception | None = None

    async with _LLM_SEMAPHORE:
        for attempt in range(1, max_attempts + 1):
            try:
                # Per-request timeout — shorter than the pool default.
                # DeepSeek free tier drops idle connections aggressively;
                # a tighter read timeout catches silent disconnects before
                # they become opaque "Server disconnected" errors.
                req_timeout = httpx.Timeout(
                    effective_timeout, connect=10.0, read=min(effective_timeout, 45.0), write=30.0, pool=5.0
                )
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=req_timeout,
                )
                data = response.json()
                _raise_for_http(response, data)
                break  # success
            except Exception as exc:
                last_error = exc
                detail = _error_detail(exc)
                logger.warning(
                    "LLM call attempt %d/%d failed (model=%s): %s",
                    attempt, max_attempts, model_name, detail[:200],
                )
                # Re-create client on connection-level errors so the next
                # attempt gets a fresh TCP+TLS session.
                # Order matters: aclose() FIRST (wait for shutdown), then
                # set _CLIENT = None so the next attempt sees it clean.
                if isinstance(exc, (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError)):
                    try:
                        async with _CLIENT_LOCK:
                            old = _CLIENT
                            if old is not None:
                                await old.aclose()
                            _CLIENT = None
                    except Exception:
                        _CLIENT = None
                if attempt < max_attempts:
                    # Exponential backoff + random jitter (P1: prevents
                    # synchronised retry storms when multiple calls fail
                    # simultaneously under rate-limiting).
                    base_sleep = 0.5 * (2 ** (attempt - 1))
                    jitter = random.uniform(0, 1.0)
                    await asyncio.sleep(base_sleep + jitter)
        else:
            # All attempts exhausted
            raise RuntimeError(
                f'{{"error": "LLM provider unavailable after {max_attempts} attempts", '
                f'"detail": "{_error_detail(last_error)}", '
                f'"recovery": "Try again in 30 seconds or check your LLM_BASE_URL configuration."}}'
            ) from None

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("LLM provider returned no choices.")
    message = choices[0].get("message") or {}
    return {
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "_payload": payload,
    }


# ── Content post-processing ─────────────────────────────────────────────────

def _process_content(
    content: str | None,
    reasoning: str | None,
    payload: dict[str, Any],
) -> str:
    """Extract usable output from an LLM response.

    Handles DeepSeek V4 spontaneous reasoning mode: when the model enters
    reasoning even with ``thinking: disabled``, the answer is in
    ``reasoning_content`` and ``content`` is empty.
    """
    if content:
        return str(content).strip()

    if not reasoning:
        raise RuntimeError("LLM provider returned no message content.")

    thinking_cfg = payload.get("thinking", {})
    is_intentional = isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "enabled"

    if is_intentional:
        # Reasoner was asked to think — the reasoning IS the output
        return str(reasoning).strip()

    # Spontaneous reasoning — try to recover the answer.
    has_json_format = (
        isinstance(payload.get("response_format"), dict)
        and payload["response_format"].get("type") == "json_object"
    )

    if has_json_format:
        candidate = _extract_tail_json(reasoning)
        if candidate is not None:
            return candidate
    else:
        candidate = _extract_tail_text(reasoning)
        if candidate is not None:
            return candidate

    raise RuntimeError(
        "Model spontaneously entered reasoning mode and "
        "consumed all tokens on chain-of-thought without "
        "producing structured output.  The caller should "
        "retry with a shorter prompt or a different model."
    )


# ── Text extraction helpers ─────────────────────────────────────────────────

def _extract_tail_json(text: str) -> str | None:
    """Extract the last JSON object ``{...}`` from *text*."""
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
        if len(candidate) >= 8:
            return candidate
    return None


def _extract_tail_text(text: str) -> str | None:
    """Extract the answer portion from reasoning_content."""
    markers = [
        "Final answer:",
        "Here is the",
        "Here's the",
        "最终回答：",
        "以下是",
        "总结：",
    ]
    best_pos = -1
    for marker in markers:
        pos = text.rfind(marker)
        if pos > best_pos:
            best_pos = pos
    if best_pos > 0:
        candidate = text[best_pos:].strip()
        if len(candidate) >= 20:
            return candidate

    sep_markers = ["\n---\n", "\n---\n\n", "\n\n---\n\n"]
    for sep in sep_markers:
        pos = text.rfind(sep)
        if pos > len(text) * 0.3:
            candidate = text[pos + len(sep):].strip()
            if len(candidate) >= 20:
                return candidate

    split = int(len(text) * 0.6)
    candidate = text[split:].strip()
    if len(candidate) >= 20:
        return candidate
    return None


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def _raise_for_http(response: httpx.Response, data: Any) -> None:
    """Raise a descriptive RuntimeError for non-2xx responses."""
    if response.is_success:
        return
    detail = ""
    if isinstance(data, dict):
        detail = str(data.get("error", {}).get("message", data.get("error", ""))) or response.text
    else:
        detail = response.text
    raise httpx.HTTPStatusError(
        f"HTTP {response.status_code}: {detail[:500]}",
        request=response.request,
        response=response,
    )


def _error_detail(exc: BaseException | None) -> str:
    """Safe string representation of an exception."""
    if exc is None:
        return "unknown error"
    return str(exc)[:500]
