"""Lightweight Tool protocol for CompIntel data sources.

A Tool is a named callable with a JSON Schema parameter spec and an
optional ``enabled`` predicate.  This protocol makes data sources
pluggable: adding a new source means implementing one async function
and registering it in ``compintel.tools.registry`` — no changes to
graph.py or agent classes required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

#: Signature for async tool execution functions.
ExecuteFn = Callable[..., Awaitable[dict[str, Any]]]

#: Signature for optional enabled-check predicates.
EnabledPredicate = Callable[[dict[str, Any]], bool]


@dataclass(slots=True)
class Tool:
    """A pluggable data-source tool.

    Parameters
    ----------
    name:
        Unique short identifier (e.g. ``"web_search"``).
    description:
        Human-readable one-liner shown in tool listings.
    parameters:
        JSON Schema describing the input parameters.
    execute:
        Async callable that receives ``**params`` and returns a result dict.
    enabled:
        Optional predicate — when provided and returning ``False``, the
        tool is skipped for the current invocation.  ``None`` means
        always enabled.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: ExecuteFn
    enabled: EnabledPredicate | None = None
