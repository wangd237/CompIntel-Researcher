"""Typed state reader for CompIntel Research.

Wraps the raw ``CompIntelState`` dict so agents never need to sprinkle
``isinstance(state, dict)`` guards or ``.get()`` chains with defaults.
"""

from __future__ import annotations

from typing import Any


class StateAdapter:
    """Typed wrapper around the raw ``CompIntelState`` dict.

    Every property provides a safe, type-consistent read with a sensible
    default.  Agents access ``adapter.query`` instead of
    ``state.get("query", "")``, eliminating isinstance guards and dict-key
    typos.

    Parameters
    ----------
    state:
        The raw state dict (or any object — non-dict values are treated as
        an empty state).
    """

    __slots__ = ("_state",)

    def __init__(self, state: dict[str, Any] | Any) -> None:
        self._state: dict[str, Any] = state if isinstance(state, dict) else {}

    # ── core identity ──────────────────────────────────────────────────

    @property
    def query(self) -> str:
        """The original user query string."""
        return str(self._state.get("query", ""))

    @property
    def language(self) -> str:
        """Detected language code: ``"zh"`` or ``"en"``."""
        return str(self._state.get("language", "en"))

    # ── intent parsing outputs ─────────────────────────────────────────

    @property
    def intent(self) -> dict[str, Any]:
        """Parsed intent analysis result."""
        val = self._state.get("intent")
        return val if isinstance(val, dict) else {}

    @property
    def target(self) -> str:
        """Primary company being analysed."""
        return str(self._state.get("target") or self.intent.get("target") or "")

    @property
    def market_segment(self) -> str:
        """Industry / market category inferred from the query."""
        return str(self._state.get("market_segment") or self.intent.get("market_segment") or "")

    @property
    def competitors(self) -> list[dict[str, Any]]:
        """List of competitor descriptors extracted from the query."""
        val = self._state.get("competitors")
        return val if isinstance(val, list) else []

    @property
    def research_questions(self) -> list[str]:
        """Research questions generated during intent analysis."""
        val = self._state.get("research_questions")
        if isinstance(val, list):
            return [str(item) for item in val if str(item).strip()]
        return []

    # ── planning ───────────────────────────────────────────────────────

    @property
    def research_plan(self) -> dict[str, Any]:
        """Structured research plan keyed by competitor name."""
        val = self._state.get("research_plan")
        return val if isinstance(val, dict) else {}

    # ── profiling ──────────────────────────────────────────────────────

    @property
    def profiles(self) -> list[dict[str, Any]]:
        """Aggregated competitor profiles (prefer curated after curator runs)."""
        # curator output takes priority — it has cleaned + graded profiles
        val = self._state.get("curated_profiles")
        if isinstance(val, list) and val:
            return val
        val = self._state.get("profiles")
        if isinstance(val, list):
            return val
        # Fall back to competitor_profiles (aliased in older state shapes)
        val = self._state.get("competitor_profiles")
        return val if isinstance(val, list) else []

    @property
    def competitor_profiles(self) -> list[dict[str, Any]]:
        """Alias for profiles (used in some state shapes)."""
        return self.profiles

    # ── analysis ───────────────────────────────────────────────────────

    @property
    def market_analysis(self) -> dict[str, Any]:
        """Aggregated market landscape from the market analyst."""
        val = self._state.get("market_analysis")
        return val if isinstance(val, dict) else {}

    @property
    def swot_analysis(self) -> dict[str, Any]:
        """SWOT matrix from the SWOT synthesizer."""
        val = self._state.get("swot_analysis")
        return val if isinstance(val, dict) else {}

    # ── report ─────────────────────────────────────────────────────────

    @property
    def report(self) -> dict[str, Any]:
        """Final report payload from the report writer."""
        val = self._state.get("report")
        return val if isinstance(val, dict) else {}

    @property
    def review_feedback(self) -> dict[str, Any]:
        """Reviewer feedback including score, approval, and issues."""
        val = self._state.get("review_feedback")
        return val if isinstance(val, dict) else {}

    @property
    def retry_count(self) -> int:
        """Number of review-revise cycles so far."""
        try:
            return int(self._state.get("retry_count", 0))
        except (TypeError, ValueError):
            return 0

    # ── operational metadata ───────────────────────────────────────────

    @property
    def warnings(self) -> list[str]:
        """Non-fatal warnings accumulated during execution."""
        val = self._state.get("warnings")
        if isinstance(val, list):
            return [str(item) for item in val if str(item).strip()]
        return []

    @property
    def execution_log(self) -> list[dict[str, Any]]:
        """Per-node execution events (annotated with operator.add)."""
        val = self._state.get("execution_log")
        return val if isinstance(val, list) else []

    @property
    def errors(self) -> list[dict[str, Any]]:
        """Error details accumulated during execution."""
        val = self._state.get("errors")
        return val if isinstance(val, list) else []

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Chat-style message log (reserved for future chat interface)."""
        val = self._state.get("messages")
        return val if isinstance(val, list) else []

    @property
    def status(self) -> str:
        """High-level pipeline status."""
        return str(self._state.get("status", ""))

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata attached to the pipeline run."""
        val = self._state.get("metadata")
        return val if isinstance(val, dict) else {}

    # ── execution tracker fields ───────────────────────────────────────

    @property
    def phase(self) -> str:
        """Current execution phase identifier."""
        return str(self._state.get("phase", ""))

    @property
    def phase_status(self) -> str:
        """Status of the current phase."""
        return str(self._state.get("phase_status", ""))

    @property
    def phase_owner(self) -> str:
        """Owner assigned to the current phase."""
        return str(self._state.get("phase_owner", ""))

    @property
    def phase_started_at(self) -> str:
        """ISO timestamp when the current phase started."""
        return str(self._state.get("phase_started_at", ""))

    @property
    def phase_updated_at(self) -> str:
        """ISO timestamp when the current phase was last updated."""
        return str(self._state.get("phase_updated_at", ""))

    @property
    def next_action(self) -> str:
        """Recommended next action in the pipeline."""
        return str(self._state.get("next_action", ""))

    @property
    def blockers(self) -> list[str]:
        """Active blockers preventing pipeline progress."""
        val = self._state.get("blockers")
        if isinstance(val, list):
            return [str(item) for item in val if str(item).strip()]
        return []

    @property
    def audit_notes(self) -> list[str]:
        """Notes collected for the audit trail."""
        val = self._state.get("audit_notes")
        if isinstance(val, list):
            return [str(item) for item in val if str(item).strip()]
        return []

    # ── raw access (escape hatch) ──────────────────────────────────────

    @property
    def raw(self) -> dict[str, Any]:
        """Direct access to the underlying state dict (escape hatch)."""
        return self._state

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style access for keys not covered by typed properties."""
        return self._state.get(key, default)

    def __repr__(self) -> str:
        return f"StateAdapter(query={self.query!r}, target={self.target!r})"
