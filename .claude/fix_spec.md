# Fix Specification: DeepSeek V4 Reasoning + Performance

## Context

This project (CompIntel Research) is a LangGraph-based competitive intelligence pipeline with 8+ agents. It uses DeepSeek V4 models (v4-pro, v4-flash) via OpenAI-compatible API. The two-phase reason→format pipeline, defensive compression bridge, and spontaneous reasoning handling all contribute to two problems:

1. **Runtime too long** (~6.5 min for 3 competitors)
2. **Frequent "spontaneous reasoning" errors** in narrative generation

The LLM client is in `compintel/llm.py`, the service layer in `compintel/core/llm_service.py`, and narrative generation in `compintel/agents/report_writer.py`.

---

## PROBLEM 1: Runtime Too Long

### Root Cause 1.1: Compression bridge is an extra LLM call per reasoning task

`llm_service.py:548-608` `_compress_reasoning()` — after the reasoning model produces chain-of-thought, this method makes ANOTHER LLM call to compress it. For N competitors: N (per-competitor SWOT) + 1 (cross-analysis) + 1 (reviewer) = N+2 extra calls, each ~5-10s.

**Fix:** Replace the LLM-based compression with a local extraction function. Extract bullet points, numbered lists, and lines containing key claim indicators (冒号分隔的事实陈述, "key insight:", "therefore", "结论:", etc.). This is <1ms instead of 5-10s. Keep the existing LLM compression as a fallback only when the local extraction produces <20 chars.

### Root Cause 1.2: Timeout too high (90s default)

`settings.py:29`: `llm_timeout_seconds: float = 90.0`. A stuck call blocks for 90s × 3 retries = 270s max.

**Fix:** Reduce default to 45s. Most calls complete in 5-15s. The retry mechanism handles transient slowness.

### Root Cause 1.3: Two-phase pipeline overused

`call_with_reasoning()` is used for every SWOT call and reviewer call. For simpler tasks (short profiles, thin data), a single call with `thinking=enabled` and JSON extraction from reasoning tail would suffice.

**Fix:** Add a fast-path heuristic in `call_with_reasoning()`: if the prompt is <1500 chars AND `reasoning_max_tokens` <= 1500, try a single-call approach first (thinking=enabled, extract JSON from reasoning_content tail). If that fails, fall back to the full two-phase pipeline. This saves one LLM call per simple SWOT task.

### Root Cause 1.4: Narrative semaphore too conservative

`report_writer.py:270`: `asyncio.Semaphore(3)`. For 5+ competitors, this creates unnecessary queuing.

**Fix:** Bump to 5. DeepSeek's free tier typically allows 5 concurrent connections.

---

## PROBLEM 2: Spontaneous Reasoning Mode

### Root Cause 2.1: Non-JSON tasks have no recovery path

The error "Model spontaneously entered reasoning mode and consumed all tokens on chain-of-thought without producing structured output" originates in `llm.py:195-200`. The key logic:

```python
if not content and reasoning:
    content = _extract_tail_json(reasoning)
    if content is None:
        thinking_cfg = payload.get("thinking", {})
        if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "enabled":
            content = reasoning  # intentional reasoning
        else:
            raise RuntimeError("Model spontaneously entered reasoning mode...")
```

`_extract_tail_json()` searches for `{...}` — but narrative generation produces **plain markdown, not JSON**. So when a narrative call hits spontaneous reasoning, it ALWAYS fails.

**Fix A:** When `response_format` is NOT `json_object` (i.e., this is a free-text task), treat `reasoning_content` as valid output — extract the last paragraph/section after the chain-of-thought markers. Add a method `_extract_tail_text()` that:
1. Finds common reasoning-to-answer transition markers ("Final answer:", "Here is the", "---", double newline after a concluding sentence)
2. Returns the text after the last such marker
3. Falls back to the last 40% of reasoning_content

**Fix B:** When spontaneous reasoning is detected AND `response_format` is `json_object`, retry once with escalated `max_tokens` (2x) so the reasoning has room to complete AND produce the JSON output.

### Root Cause 2.2: Prompt length triggers reasoning

Documented in `llm_service.py:517-518`: "V4 models enter reasoning more readily when the user message exceeds ~2000 characters."

Narrative prompts (`report_writer.py:326-335`) include profile summary (400 chars) + search snippets (3×200) + RAG context (2×200) + sources (3 URLs) + instruction text = easily 1800+ chars.

**Fix:** Add a prompt truncation guard in `LLMService.call()`: if `thinking` is `{"type": "disabled"}` AND prompt > 2000 chars AND `response_format` is NOT `json_object`, trim the data portions (search snippets, RAG context) to keep total <1800 chars. Log a warning when truncation occurs.

### Root Cause 2.3: temperature=0.0 triggers reasoning

Documented in `llm_service.py:515-516`: "temperature=0.0 correlates with reasoning-mode entrance on some DeepSeek V4 deployments."

**Fix:** In `call_with_reasoning()` formatting phase (line 452) and in the compression bridge call (line 594), change `temperature=0.0` to `temperature=0.1`. In `call_and_parse()`, ensure temperature is never exactly 0.0 — use 0.05 minimum.

### Root Cause 2.4: No model fallback in `create_chat_completion`

When `create_chat_completion` hits spontaneous reasoning, it raises immediately. Unlike `call_with_reasoning()` which has a smart-model fallback, the basic call path has no recovery.

**Fix:** In `_post_chat_completion()`, when spontaneous reasoning is detected:
1. First, try the `_extract_tail_text()` approach (for free-text tasks) or `_extract_tail_json()` with escalated max_tokens (for JSON tasks)
2. If that fails, raise the error (preserving existing behavior as last resort)

---

## Files to Modify

### 1. `compintel/llm.py`

- **`_post_chat_completion()`** (~line 170-212): Add `_extract_tail_text()` fallback for non-JSON tasks. Add escalated max_tokens retry. 
- **New function `_extract_tail_text()`**: Extract the answer portion from reasoning_content by finding transition markers.
- **`create_chat_completion()`** signature: No change needed.

### 2. `compintel/core/llm_service.py`

- **`call()` method** (~line 61-120): Add prompt length guard — if prompt > 2000 chars and thinking is disabled and no response_format, trim data portions.
- **`call_and_parse()` method** (~line 122-205): Ensure temperature is never exactly 0.0 (use 0.05 minimum).
- **`call_with_reasoning()` method** (~line 291-544): Add fast-path for short prompts. Change formatting temperature 0.0→0.1. 
- **`_compress_reasoning()` method** (~line 548-608): Replace LLM-based compression with local extraction. Keep LLM compression as fallback.

### 3. `compintel/settings.py`

- **Line 29**: Change `llm_timeout_seconds: float = 90.0` to `45.0`.

### 4. `compintel/agents/report_writer.py`

- **Line 270**: Change `asyncio.Semaphore(3)` to `asyncio.Semaphore(5)`.

---

## Implementation Order

1. **`llm.py` first** — Add `_extract_tail_text()`, fix spontaneous reasoning recovery
2. **`llm_service.py` second** — Prompt guard, fast-path, local compression, temperature fix
3. **`settings.py` third** — Timeout reduction
4. **`report_writer.py` fourth** — Semaphore increase

## Success Criteria

After fixes:
1. No more "Model spontaneously entered reasoning mode" errors in narrative generation — reasoning_content is used as valid output for free-text tasks
2. Total runtime reduced by 30-50% (fewer LLM calls, lower timeout)
3. JSON-structured tasks (SWOT, reviewer) still work correctly with the two-phase pipeline
4. The escalation/fallback paths are exercised gracefully, not as errors
5. All existing degradation paths (derived, template) still work
