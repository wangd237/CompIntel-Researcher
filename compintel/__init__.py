"""CompIntel Research.

Core contract modules live at the package root:
`state.py`, `schemas.py`, and `graph.py`.
Functional areas stay in subpackages.
"""

from .graph import CompIntelGraph
from .bundle import BundleWriter
from .bundle import generate_delivery_bundle
from .execution import CompIntelExecution
from .events import CompIntelEvent
from .parsing import extract_json_candidates, load_repaired_json, safe_json_dumps
from .settings import CompIntelSettings
from .schemas import (
    CompetitorCandidate,
    CompIntelAnalyzeRequest,
    CompIntelAnalyzeResponse,
    ExecutionCheckpoint,
    ExecutionTrackerSnapshot,
    IntentAnalysisResponse,
)
from .state import CompIntelState, CompetitorProfilerState
from .tracker import ExecutionTracker

__all__ = [
    "CompIntelGraph",
    "CompIntelExecution",
    "BundleWriter",
    "generate_delivery_bundle",
    "CompIntelEvent",
    "CompIntelState",
    "CompetitorCandidate",
    "CompetitorProfilerState",
    "CompIntelAnalyzeRequest",
    "CompIntelAnalyzeResponse",
    "ExecutionCheckpoint",
    "ExecutionTracker",
    "ExecutionTrackerSnapshot",
    "IntentAnalysisResponse",
    "extract_json_candidates",
    "load_repaired_json",
    "CompIntelSettings",
    "safe_json_dumps",
]
