"""Unified LLM service for CompIntel Research.

Single entry point for all LLM calls. Handles import, model resolution,
call, retry, JSON parsing, and three-tier degradation (LLM -> derived -> template).

This module is additive — it imports from ``compintel.llm`` and
``compintel.parsing`` internally so existing agent code continues to work
unchanged during the phased migration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from compintel.llm import _split_provider_model, create_chat_completion, create_chat_completion_raw
from compintel.parsing import load_repaired_json
from compintel.settings import CompIntelSettings

logger = logging.getLogger(__name__)

_MODEL_KEY_MAP: dict[str, str] = {
    "fast": "fast_llm",
    "smart": "smart_llm",
    "strategic": "strategic_llm",
    "reasoning": "reasoning_llm",
    "formatting": "formatting_llm",
}


class LLMService:
    """Single entry point for all LLM calls.

    Resolves symbolic model keys (``"fast"``, ``"smart"``, ``"strategic"``)
    to concrete model strings from settings, calls the provider, retries on
    network failure, parses JSON responses, and implements the three-tier
    degradation chain used by every CompIntel agent.

    Parameters
    ----------
    settings:
        CompIntel settings.  When *None* the service lazily loads
        ``CompIntelSettings.from_env()`` on first use.
    """

    def __init__(self, settings: CompIntelSettings | None = None) -> None:
        self._settings = settings
        self._completion_fn = None  # reserved for test injection in Phase 2
        # Deterministic call cache: maps (prompt_hash, model_key, max_tokens)
        # → parsed dict.  Only populated for temperature < 0.05 where
        # outputs are effectively deterministic.  Cleared each query via
        # clear_cache() — call this before starting a new pipeline run.
        self._call_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def clear_cache(self) -> None:
        """Clear the deterministic call cache. Call before each new query."""
        self._call_cache.clear()

    # ── properties ─────────────────────────────────────────────────────

    @property
    def settings(self) -> CompIntelSettings:
        if self._settings is None:
            self._settings = CompIntelSettings.from_env()
        return self._settings

    # ── public API ──────────────────────────────────────────────────────

    async def call(
        self,
        prompt: str,
        model_key: str = "smart",
        max_tokens: int = 1000,
        temperature: float = 0.2,
        timeout: float | None = None,
        system_prompt: str | None = None,
        thinking: dict[str, str] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Single LLM call with network retry.  Returns raw text.

        Parameters
        ----------
        prompt:
            The user message content.
        model_key:
            One of ``"fast"``, ``"smart"``, ``"strategic"`` — maps to the
            corresponding setting (``FAST_LLM``, ``SMART_LLM``,
            ``STRATEGIC_LLM``).
        max_tokens:
            Maximum tokens in the completion.
        temperature:
            Sampling temperature (0.0–2.0).  DeepSeek marks this as
            "will not be validated" in thinking mode — use 1.0-1.5
            in thinking, 0.2 in non-thinking.
        timeout:
            Override for the global ``LLM_TIMEOUT_SECONDS`` setting.
        system_prompt:
            Optional system-level instruction prepended to the messages list.
        thinking:
            DeepSeek V4 thinking control dict.  ``{"type": "disabled"}``
            for structured-output calls; ``{"type": "enabled"}`` for
            reasoning-heavy tasks.
        response_format:
            Optional OpenAI-compatible response format dict, e.g.
            ``{"type": "json_object"}``.
        """
        model = self._resolve_model(model_key)
        provider, model_name = _split_provider_model(model)

        # P2-1: Guard against prompt length triggering spontaneous reasoning.
        # DeepSeek V4 models enter reasoning mode more readily when the user
        # message exceeds ~2000 characters, even with thinking=disabled.
        # For free-text tasks without JSON output, trim the prompt to stay
        # under the threshold.
        is_free_text = (
            response_format is None
            or response_format.get("type") != "json_object"
        )
        thinking_disabled = (
            thinking is not None
            and thinking.get("type") == "disabled"
        )
        if is_free_text and thinking_disabled and len(prompt) > 2000:
            # Trim from the middle (data sections), preserving instructions
            # at the start and end.
            head = prompt[:1200]
            tail = prompt[-400:]
            prompt = f"{head}\n…[trimmed to prevent reasoning mode]…\n{tail}"
            logger.debug("Trimmed prompt from >2000 to %d chars to prevent spontaneous reasoning", len(prompt))

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": model_name,
            "llm_provider": provider,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        if thinking is not None:
            kwargs["thinking"] = thinking
        if response_format is not None:
            kwargs["response_format"] = response_format
        return await create_chat_completion(**kwargs)

    async def call_and_parse(
        self,
        prompt: str,
        model_key: str = "smart",
        max_tokens: int = 1000,
        temperature: float = 0.2,
        timeout: float | None = None,
        system_prompt: str | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any] | None:
        """Call LLM with retry, then parse JSON.

        Retries both network failures AND unparseable responses up to
        *max_attempts* times with exponential back-off on network errors
        and fixed 1 s back-off on parse failures.

        Returns
        -------
        dict or None
            Parsed JSON dict on success, ``None`` after exhausting all
            attempts.
        """
        # ── Deterministic cache (Phase 3) ─────────────────────────────
        # When temperature is very low (< 0.05), identical prompt + model
        # + max_tokens produces identical output.  Cache hits eliminate
        # LLM calls entirely — most useful during Reviewer revise loops
        # and repeated calls with the same profile data.
        if temperature < 0.05:
            import hashlib
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            cache_key: tuple[str, str, int] = (prompt_hash, model_key, max_tokens)
            cached = self._call_cache.get(cache_key)
            if cached is not None:
                logger.debug("Deterministic cache hit for prompt hash %s", prompt_hash)
                return dict(cached)  # shallow copy so callers can't mutate cached value
        else:
            cache_key = None

        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw = await self.call(
                    prompt=prompt,
                    model_key=model_key,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 0.05),  # P2: 0.0 triggers reasoning on V4
                    timeout=timeout,
                    system_prompt=system_prompt,
                    thinking={"type": "disabled"},
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt, max_attempts, exc,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                continue

            parsed = load_repaired_json(str(raw))
            if isinstance(parsed, dict):
                if cache_key is not None:
                    self._call_cache[cache_key] = dict(parsed)
                return parsed
            # DeepSeek sometimes wraps the JSON object in a single-element list
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                logger.debug("LLM returned a single-element list; unwrapping to dict")
                if cache_key is not None:
                    self._call_cache[cache_key] = dict(parsed[0])
                return parsed[0]

            # Empty response: treat like a network failure so the retry loop
            # has a chance to get a real answer on the next attempt.
            raw_str = str(raw).strip()
            if not raw_str:
                logger.warning(
                    "LLM returned empty response (attempt %d/%d, model_key=%s) — retrying",
                    attempt, max_attempts, model_key,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(1)
                continue

            # Diagnostic: log first/last 200 chars of raw output so we can
            # see *what* pattern the LLM produced that failed to parse.
            preview = raw_str[:200] + ("…" if len(raw_str) > 200 else "")
            tail = raw_str[-120:] if len(raw_str) > 200 else ""
            logger.warning(
                "LLM returned unparseable JSON (attempt %d/%d) — "
                "raw_len=%d preview=%r tail=%r model_key=%s",
                attempt, max_attempts, len(raw_str), preview, tail, model_key,
            )
            if attempt < max_attempts:
                await asyncio.sleep(1)

        if last_error is not None:
            logger.error(
                "LLM parsing failed after %d attempts (last error: %s)",
                max_attempts, last_error,
            )
        return None

    async def call_with_degradation(
        self,
        prompt: str,
        model_key: str = "smart",
        max_tokens: int = 1000,
        temperature: float = 0.2,
        derived_fn: Callable[[], dict[str, Any]] | None = None,
        template_fn: Callable[[], dict[str, Any]] | None = None,
        timeout: float | None = None,
        system_prompt: str | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Three-tier degradation chain: LLM -> derived -> template.

        **Tier 1 — LLM (best):**
        Calls the LLM with retry + JSON parse.  Returns the parsed dict on
        success.

        **Tier 2 — derived (good):**
        When the LLM is configured (API key present) but fails after all
        retries, *derived_fn* is called.  This typically performs local
        computation (e.g. extracting trends from search snippets) that
        produces better output than a bare template.

        **Tier 3 — template (minimum):**
        When no LLM API key is configured, *template_fn* is called.  This
        returns a minimal placeholder payload that keeps the pipeline
        running without blocking on missing credentials.

        Parameters
        ----------
        prompt:
            LLM prompt (ignored when no API key is present).
        derived_fn:
            Called when LLM is configured but fails.  Must return a dict.
        template_fn:
            Called when no LLM API key is present.  Must return a dict.
        (other parameters are forwarded to :meth:`call_and_parse`).

        Returns
        -------
        dict
            Never ``None`` — at least the template result is returned.
        """
        # Tier 3: no API key at all → template
        if not self.settings.llm_api_key:
            if template_fn is not None:
                return template_fn()
            # Ultimate fallback when even the template is missing
            return {
                "_degraded": True,
                "_tier": "template",
                "note": "LLM API key not configured and no template provided.",
            }

        # Tier 1: try LLM
        try:
            parsed = await self.call_and_parse(
                prompt=prompt,
                model_key=model_key,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                system_prompt=system_prompt,
                max_attempts=max_attempts,
            )
            if parsed is not None:
                return parsed
        except Exception as exc:
            logger.warning("LLM call raised exception in degradation chain: %s", exc)

        # Tier 2: LLM failed but was configured → derived
        if derived_fn is not None:
            return derived_fn()

        # Derived also missing → minimal template
        if template_fn is not None:
            return template_fn()
        return {
            "_degraded": True,
            "_tier": "derived",
            "note": "LLM call failed and no derived function provided.",
        }

    async def call_with_reasoning(
        self,
        prompt: str,
        *,
        reasoning_model_key: str = "reasoning",
        formatting_model_key: str = "formatting",
        reasoning_max_tokens: int | None = None,
        formatting_max_tokens: int | None = None,
        temperature: float = 0.2,
        timeout: float | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any] | None:
        """Two-phase reason→format pipeline for structured-output tasks.

        **Phase 1 — Reasoning:**
        Calls the reasoning model (e.g. ``deepseek-reasoner``) to produce
        deep chain-of-thought analysis.  The FULL reasoning text is captured —
        not just the tail JSON — so the formatting phase has rich context.

        **Phase 2 — Formatting:**
        Injects the reasoning text as context and asks the formatting model
        (e.g. ``deepseek-chat``) to produce ONLY a strict JSON object.

        Both phases have independent retry with exponential back-off.
        If the reasoner fails after all attempts, ``None`` is returned and
        the caller's existing degradation chain (derived → template) takes over.

        Parameters
        ----------
        prompt:
            The original task prompt (same format as ``call_and_parse``).
        reasoning_model_key:
            Symbolic key for the reasoning model.  Default ``"reasoning"``
            maps to ``REASONING_LLM`` (deepseek-reasoner).
        formatting_model_key:
            Symbolic key for the formatting model.  Default ``"formatting"``
            maps to ``FORMATTING_LLM`` (deepseek-chat).
        reasoning_max_tokens:
            Max tokens for the reasoning phase.  Defaults to
            ``settings.reasoning_max_tokens`` (4000).
        formatting_max_tokens:
            Max tokens for the formatting phase.  Defaults to
            ``settings.formatting_max_tokens`` (2000).
        temperature:
            Used for the reasoning phase; formatting uses 0.0 for determinism.
        timeout:
            Forwarded to both calls.
        max_attempts:
            Max retries for each phase independently.

        Returns
        -------
        dict or None
            Parsed JSON on success, ``None`` if either phase fails after all
            retries.
        """
        if reasoning_max_tokens is None:
            reasoning_max_tokens = self.settings.reasoning_max_tokens
        if formatting_max_tokens is None:
            formatting_max_tokens = self.settings.formatting_max_tokens

        # ── Fast-path: single-call for short/simple prompts ──────────────
        # When the prompt is short and token budget is modest, try a single
        # reasoning call and extract JSON from the reasoning tail directly.
        # This saves one full formatting-phase LLM call (~5-10s) per task.
        if len(prompt) < 2500 and reasoning_max_tokens <= 3000:
            reason_model = self._resolve_model(reasoning_model_key)
            reason_provider, reason_model_name = _split_provider_model(reason_model)
            try:
                raw = await create_chat_completion_raw(
                    messages=[{"role": "user", "content": prompt}],
                    model=reason_model_name,
                    llm_provider=reason_provider,
                    max_tokens=reasoning_max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    thinking={"type": "enabled"},
                )
                reasoning = (raw.get("reasoning_content") or raw.get("content") or "").strip()
                if reasoning:
                    # Try to extract JSON from the reasoning tail
                    from compintel.llm import _extract_tail_json
                    json_candidate = _extract_tail_json(reasoning)
                    if json_candidate:
                        parsed = load_repaired_json(json_candidate)
                        if isinstance(parsed, dict):
                            logger.debug("Fast-path reasoning succeeded (single call)")
                            return parsed
                        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                            logger.debug("Fast-path reasoning succeeded (single call, unwrapped list)")
                            return parsed[0]
            except Exception:
                pass  # Fast-path failed — fall through to full two-phase pipeline

        # ── Phase 1: Reasoning ──────────────────────────────────────────
        reason_model = self._resolve_model(reasoning_model_key)
        reason_provider, reason_model_name = _split_provider_model(reason_model)

        reason_messages: list[dict[str, str]] = [
            {"role": "user", "content": prompt},
        ]

        reasoning_text: str = ""
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw = await create_chat_completion_raw(
                    messages=reason_messages,
                    model=reason_model_name,
                    llm_provider=reason_provider,
                    max_tokens=reasoning_max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    thinking={"type": "enabled"},
                )
                reasoning_text = (raw.get("reasoning_content") or raw.get("content") or "").strip()
                if reasoning_text:
                    break  # got useful reasoning
                logger.warning(
                    "Reasoner returned empty reasoning (attempt %d/%d, model=%s)",
                    attempt, max_attempts, reason_model_name,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Reasoner call failed (attempt %d/%d): %s",
                    attempt, max_attempts, str(exc)[:200],
                )
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 4))

        if not reasoning_text:
            logger.error(
                "Reasoner failed after %d attempts (last_error=%s)",
                max_attempts, last_error,
            )
            return None

        logger.debug("Reasoner produced %d chars of analysis", len(reasoning_text))

        # ── Compression bridge (Claude Code-inspired) ──────────────────
        # Raw chain-of-thought is model-internal dialogue — passing it
        # verbatim to the formatter bloats the prompt to 3000+ tokens,
        # which can trigger reasoning behaviour even in non-reasoning
        # models (deepseek-v4-flash).  The fix: extract only the KEY
        # FACTUAL CLAIMS and STRATEGIC CONCLUSIONS from the reasoning,
        # discard the "let me think about..." scaffolding.
        #
        # This is analogous to how Claude Code separates unstructured
        # <thinking> blocks from structured tool-call output — the
        # downstream consumer never sees the raw reasoning, only the
        # distilled conclusions.
        reasoning_text = await self._compress_reasoning(
            reasoning_text, formatting_model_key, temperature,
        )

        # ── Phase 2: Formatting ──────────────────────────────────────────
        format_model = self._resolve_model(formatting_model_key)
        format_provider, format_model_name = _split_provider_model(format_model)

        # The compressed reasoning is ~300-500 chars of structured claims,
        # not 4000+ chars of raw chain-of-thought.  This keeps the formatter
        # prompt under ~800 tokens — well below the threshold where
        # non-reasoning models start burning tokens on "how to format this".
        format_prompt = (
            f"{prompt}\n\n"
            f"--- Key Analysis Findings ---\n"
            f"{reasoning_text}\n"
            f"--- End Findings ---\n\n"
            f"Based on the above findings, output ONLY a valid JSON object. "
            f"Do NOT include any explanation, markdown fences, or commentary — "
            f"return pure JSON starting with {{ and ending with }}."
        )

        format_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a JSON formatting assistant. Your ONLY output is "
                    "a valid JSON object. Never include markdown fences, "
                    "commentary, or any text outside the JSON."
                ),
            },
            {"role": "user", "content": format_prompt},
        ]

        for attempt in range(1, max_attempts + 1):
            try:
                raw = await create_chat_completion(
                    messages=format_messages,
                    model=format_model_name,
                    llm_provider=format_provider,
                    max_tokens=formatting_max_tokens,
                    temperature=0.1,  # P2: 0.0 triggers reasoning on V4
                    timeout=timeout,
                    thinking={"type": "disabled"},
                    response_format={"type": "json_object"},
                )
                parsed = load_repaired_json(str(raw))
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                    logger.debug("Formatter returned single-element list; unwrapping")
                    return parsed[0]

                raw_str = str(raw).strip()
                if not raw_str:
                    logger.warning(
                        "Formatter returned empty (attempt %d/%d, model=%s)",
                        attempt, max_attempts, format_model_name,
                    )
                else:
                    preview = raw_str[:200] + ("…" if len(raw_str) > 200 else "")
                    logger.warning(
                        "Formatter returned unparseable JSON (attempt %d/%d) "
                        "len=%d preview=%r",
                        attempt, max_attempts, len(raw_str), preview,
                    )
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                is_spontaneous_reasoning = (
                    "spontaneously entered reasoning" in err_str
                    or "consumed all tokens on chain-of-thought" in err_str
                )
                logger.warning(
                    "Formatter call failed (attempt %d/%d): %s",
                    attempt, max_attempts, err_str[:200],
                )
                if is_spontaneous_reasoning:
                    # Don't retry — the model will do this again on
                    # every attempt.  Break out and fall through to
                    # the smart-model fallback immediately.
                    logger.warning(
                        "Formatter hit spontaneous reasoning — "
                        "breaking out of retry loop (attempt %d/%d)",
                        attempt, max_attempts,
                    )
                    break
            if attempt < max_attempts:
                await asyncio.sleep(1)

        logger.error(
            "Formatter failed after %d attempts (last_error=%s) — "
            "reasoning→format pipeline degraded, falling back to "
            "single-model structured-output call",
            max_attempts, last_error,
        )

        # ── Fallback: switch model and drop reasoning context ──────────
        # When the formatting model (v4-flash) burns all tokens on a
        # spontaneous chain-of-thought, retrying the same model with the
        # same prompt shape is futile.  Two changes:
        #
        # 1. Switch to the SMART model (v4-pro) which is less prone to
        #    uncontrolled reasoning on structured-output tasks.
        # 2. Use a small non-zero temperature — temperature=0.0 correlates
        #    with reasoning-mode entrance on some DeepSeek V4 deployments.
        # 3. Trim the prompt if it is very long — V4 models enter reasoning
        #    more readily when the user message exceeds ~2000 characters.
        try:
            trimmed_prompt = prompt
            if len(prompt) > 4000:
                trimmed_prompt = prompt[:4000] + (
                    "\n\n…[truncated — see full context in reasoning log]"
                )

            fallback = await self.call_and_parse(
                prompt=trimmed_prompt,
                model_key="smart",  # v4-pro, not v4-flash
                max_tokens=formatting_max_tokens,
                temperature=0.1,
                system_prompt=(
                    "You are a JSON formatting assistant.  "
                    "Respond with ONLY a valid JSON object — no explanation, "
                    "no markdown fences, no commentary.  "
                    "Start your response with { and end with }."
                ),
            )
            if isinstance(fallback, dict):
                logger.info("Formatter fallback succeeded (smart model, no reasoning)")
                return fallback
        except Exception as fb_exc:
            logger.warning("Formatter fallback also failed: %s", str(fb_exc)[:200])

        return None

    # ── internals ───────────────────────────────────────────────────────

    async def _compress_reasoning(
        self,
        reasoning_text: str,
        formatting_model_key: str,
        temperature: float,
    ) -> str:
        """Distill raw chain-of-thought into structured claims (≤400 chars).

        Uses local extraction (<1 ms) — when that produces fewer than 20 chars
        of useful output, truncate to 800 chars directly.  The LLM-based
        compression call has been removed entirely: it was an extra LLM call
        that itself could trigger spontaneous reasoning, and local extraction
        hits the >90% case in practice.
        """
        # If reasoning is already short, skip compression.
        if len(reasoning_text) <= 800:
            return reasoning_text

        # ── Local extraction (<1 ms) ──────────────────────────────────
        local = self._extract_key_claims(reasoning_text)
        if local and len(local) >= 20:
            logger.debug(
                "Locally compressed reasoning from %d → %d chars",
                len(reasoning_text), len(local),
            )
            return local

        # ── Fallback: truncate to first ~800 chars ─────────────────────
        return reasoning_text[:800]

    @staticmethod
    def _extract_key_claims(text: str) -> str:
        """Extract key factual claims from reasoning text without an LLM call.

        Looks for:
        1. Numbered / bulleted lists (lines starting with digits or dashes that
           contain claim-like content)
        2. Lines containing key insight indicators (冒号分隔的事实陈述,
           "key insight:", "therefore", "结论:", "important:", etc.)
        3. The last few substantive sentences (as a final fallback)

        Returns a compact string of extracted claims, or empty string if nothing
        useful was found.
        """
        lines = text.split("\n")
        claims: list[str] = []

        # Pattern 1: Numbered/bulleted list items with substance
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Numbered: "1. xxx", "1) xxx", "第1"
            # Bulleted: "- xxx", "* xxx", "• xxx"
            is_list_item = False
            if stripped[0] in "-*•" or (
                len(stripped) > 2
                and (stripped[0].isdigit() and stripped[1] in ".)、")
            ):
                is_list_item = True
            if is_list_item and len(stripped) >= 15:
                claims.append(stripped)

        # Pattern 2: Lines with claim indicators
        claim_keywords = (
            "key insight:", "therefore", "结论:", "important:",
            "关键发现:", "关键洞察:", "总结:", "核心观点:",
            "优势:", "劣势:", "机会:", "威胁:",
            "strength:", "weakness:", "opportunity:", "threat:",
            "differentiator:", "competitive",
        )
        for line in lines:
            stripped = line.strip()
            low = stripped.lower()
            if any(kw in low for kw in claim_keywords):
                if stripped not in claims and len(stripped) >= 15:
                    claims.append(stripped)

        if claims:
            return "\n".join(claims[:10])

        # Pattern 3: Last few substantive sentences (80+ char lines likely
        # to be conclusions, not scaffolding)
        substantive = [l.strip() for l in lines if len(l.strip()) >= 80]
        if substantive:
            return "\n".join(substantive[-5:])

        return ""

    def _resolve_model(self, model_key: str) -> str:
        """Map a symbolic key to the concrete model string from settings."""
        settings_attr = _MODEL_KEY_MAP.get(model_key, "smart_llm")
        return getattr(self.settings, settings_attr, self.settings.smart_llm)
