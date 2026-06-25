"""JSON parsing helpers for CompIntel Research."""

from __future__ import annotations
import logging

import json
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    import json_repair
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    json_repair = None


JSON_BLOCK_PATTERNS = (
    re.compile(r"```(?:json)?\s*(?P<payload>[\s\S]*?)```", re.IGNORECASE),
    re.compile(r"(?P<payload>\{[\s\S]*\})"),
    # Also catch JSON arrays (e.g. DeepSeek sometimes wraps dicts in a list)
    re.compile(r"(?P<payload>\[[\s\S]*\])"),
)


def extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for pattern in JSON_BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            payload = match.group("payload").strip()
            if payload and payload not in seen:
                seen.add(payload)
                candidates.append(payload)
    return candidates


def _repair_truncated_json(candidate: str) -> str:
    """Pre-process DeepSeek-typical truncation errors before handing to json_repair.

    DeepSeek commonly cuts off mid-string when max_tokens is hit, or drops
    commas between adjacent string fields. The regex-based fixes here are
    safe because they only act on the final incomplete token — they never
    alter structurally valid JSON.
    """
    if not candidate or not candidate.strip():
        return candidate

    # Fast-path: if it is already valid JSON, return immediately
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        pass

    # 1. Truncate to last complete top-level object (LAST closing brace where brace_count reaches 0)
    brace_count = 0
    last_valid_pos = 0
    for i, ch in enumerate(candidate):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                last_valid_pos = i + 1
    if last_valid_pos > 0 and last_valid_pos < len(candidate) and candidate.startswith("{"):
        candidate = candidate[:last_valid_pos]

    # 2. Fix unterminated string - close incomplete "key": "value..." at end
    candidate = re.sub(
        r'("(?:\\.|[^"\\])*"\s*:\s*)"(?:\\.|[^"\\])*\Z',
        r'\1""',
        candidate,
    )

    # 3. Fix missing comma: two string values on adjacent lines without comma
    candidate = re.sub(r'"\s*\n\s*"', r'",\n  "', candidate)

    # 4. Fix missing comma between a closing brace/bracket and a following key/value
    candidate = re.sub(r'([}\]"\d])\s*\n\s*"', r'\1,\n  "', candidate)

    # 5. Fix trailing comma before closing brace or bracket
    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)

    # 6. Close unclosed braces by appending missing } characters
    brace_count = 0
    for ch in candidate:
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
    if brace_count > 0:
        candidate = candidate.rstrip() + ("}" * brace_count)

    return candidate

def load_repaired_json(text: str) -> Any:
    for candidate in [text.strip(), *extract_json_candidates(text)]:
        if not candidate:
            continue
        candidate = _repair_truncated_json(candidate)
        try:
            if json_repair is not None:
                return json_repair.loads(candidate)
            return json.loads(candidate)
        except Exception:
            preview = candidate[:180] if len(candidate) > 180 else candidate
            tail = candidate[-80:] if len(candidate) > 180 else ""
            logger.warning(
                "Failed to parse JSON candidate (len=%d, preview=%r, tail=%r)",
                len(candidate), preview, tail,
            )
            # Full candidate is available at DEBUG level for offline analysis
            logger.debug("Full unparseable candidate:\n%s", candidate)
            continue
    return None


def safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
