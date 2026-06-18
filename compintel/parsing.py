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


def load_repaired_json(text: str) -> Any:
    for candidate in [text.strip(), *extract_json_candidates(text)]:
        if not candidate:
            continue
        try:
            if json_repair is not None:
                return json_repair.loads(candidate)
            return json.loads(candidate)
        except Exception:
            logger.exception("Failed to parse JSON candidate")
            continue
    return None


def safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
