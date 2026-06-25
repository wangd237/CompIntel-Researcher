"""Prompt registry for CompIntel Research.

Each agent's prompt lives in a YAML file keyed by agent name.  The file
includes version, model parameters, and the prompt template with
``{variable}`` placeholders for runtime interpolation.

Usage::

    from compintel.prompts import load_prompt

    prompt = load_prompt("intent_analyst")
    text = prompt.format(query="分析 Notion 的竞品格局")
    result = await llm.call_and_parse(text, model_key=prompt.model_key, ...)

When a YAML file is missing, ``load_prompt`` returns a sensible fallback
so that the system never breaks on missing template files.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent

# Matches {identifier} but NOT {{escaped}} or {"json":"like"}.
# JSON objects have : or " right after { — skip those.
# Escaped {{ }} are treated as literal single braces.
_SAFE_VAR_RE = re.compile(r"(?<!\{)\{([a-zA-Z_]\w*)\}(?!\})")


class Prompt(NamedTuple):
    """A loaded prompt template with its metadata."""

    name: str
    version: str
    model_key: str
    max_tokens: int
    temperature: float
    system: str | None
    template: str

    def format(self, **kwargs: Any) -> str:
        """Interpolate ``{var}`` placeholders, leaving JSON braces alone.

        Placeholders use standard ``{name}`` syntax.  Literal JSON like
        ``{"key": "val"}`` is left untouched because the opening ``{``
        is immediately followed by ``"`` (not a valid Python identifier).

        Use ``{{`` and ``}}`` to produce literal ``{`` ``}`` when needed
        (e.g. ``{{"company": {{"name": "...", ...}} }}``).
        """
        text = self.template
        missing = [m.group(1) for m in _SAFE_VAR_RE.finditer(text)
                    if m.group(1) not in kwargs]
        if missing:
            raise KeyError(f"Missing prompt variables: {missing}")

        result = _SAFE_VAR_RE.sub(
            lambda m: str(kwargs[m.group(1)]),
            text,
        )
        # Unescape doubled braces
        result = result.replace("{{", "{").replace("}}", "}")
        return result


def load_prompt(name: str) -> Prompt:
    """Load a prompt template from its YAML file.

    Parameters
    ----------
    name:
        Prompt name without extension (e.g. ``"intent_analyst"``).

    Returns
    -------
    Prompt
        Loaded template.  If the YAML file is missing or unreadable a
        minimal fallback prompt is returned so the system stays operational.
    """
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        logger.warning("Prompt file %s not found — using fallback", path.name)
        return _fallback(name)

    try:
        return _load_yaml(name, path)
    except Exception as exc:
        logger.warning("Failed to load prompt %s: %s — using fallback", path.name, exc)
        return _fallback(name)


@lru_cache(maxsize=32)
def _load_yaml(name: str, path: Path) -> Prompt:
    """Parse a YAML prompt file.  Cached so each agent pays the I/O cost once."""
    # Inline import — yaml is stdlib in our target env (pyyaml is not a hard dep)
    try:
        import yaml as _yaml_lib
    except ImportError:
        # Fall back to a minimal inline parser for simple YAML (one-level dict,
        # string values, multiline strings with |)
        data = _parse_simple_yaml(path.read_text(encoding="utf-8"))
    else:
        with open(path, encoding="utf-8") as fh:
            data = _yaml_lib.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must be a dict, got {type(data).__name__}")

    template = str(data.get("prompt", ""))
    if not template.strip():
        logger.warning("Prompt %s has empty template", name)

    return Prompt(
        name=name,
        version=str(data.get("version", "0.0.0")),
        model_key=str(data.get("model_key", "smart")),
        max_tokens=int(data.get("max_tokens", 1000)),
        temperature=float(data.get("temperature", 0.2)),
        system=data.get("system") if data.get("system") else None,
        template=template,
    )


def _fallback(name: str) -> Prompt:
    """Minimal prompt that won't crash the pipeline."""
    return Prompt(
        name=name,
        version="fallback",
        model_key="smart",
        max_tokens=1000,
        temperature=0.2,
        system=None,
        template=(
            f"You are CompIntel's {name.replace('_', ' ')}. "
            "Produce a valid JSON response.\nQuery: {query}"
        ),
    )


# ── Minimal YAML parser for envs without pyyaml ──────────────────────────
# Handles the subset of YAML that prompt files use: top-level string keys,
# string / null values, and literal-block scalars (|).

def _parse_simple_yaml(raw: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    block_lines: list[str] = []
    in_block: bool = False

    for line in raw.split("\n"):
        # Detect block scalar continuation
        if in_block:
            if line and (line[0] == " " or line == ""):
                block_lines.append(line)
                continue
            else:
                result[current_key] = "\n".join(
                    _rstrip_trailing_newlines(block_lines)
                )
                in_block = False
                block_lines = []
                current_key = None

        stripped = line.rstrip()
        if not stripped or stripped.startswith("#"):
            continue

        # Block scalar header:  key: |
        if ":" in stripped:
            idx = stripped.index(":")
            key = stripped[:idx].strip()
            rest = stripped[idx + 1:].strip()
            if rest == "|":
                current_key = key
                in_block = True
                block_lines = []
            elif rest == "" or rest in ("null", "~"):
                result[key] = None
            else:
                value = rest.strip("\"'").strip()
                result[key] = _coerce_value(value)

    # Flush trailing block
    if in_block and current_key:
        result[current_key] = "\n".join(
            _rstrip_trailing_newlines(block_lines)
        )

    return result


def _rstrip_trailing_newlines(lines: list[str]) -> list[str]:
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _coerce_value(raw: str) -> Any:
    """Try to parse as float/int, fall back to string."""
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw
