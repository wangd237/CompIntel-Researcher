"""Regression tests for CompIntel Research.

These tests guard against prompt / pipeline changes that silently
degrade report quality on known queries.

``test_baseline_file_valid`` runs in < 1 s — suitable for pre-commit.
``test_regression_suite`` actually runs the pipeline for every query in
the test bank and should be invoked manually or in CI, not on every
commit.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Only import the test infrastructure — don't run the pipeline yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compintel.regression_test import (
    TEST_QUERIES,
    _BASELINE_PATH,
    diff,
    load_baseline,
    run_single,
)


def test_baseline_file_valid() -> None:
    """Baseline JSON must exist and must have valid structure."""
    if not _BASELINE_PATH.exists():
        pytest.skip(f"Baseline file {_BASELINE_PATH} not found — "
                    f"run python -m compintel.regression_test first")

    baseline = load_baseline()
    assert isinstance(baseline, dict), "baseline must be a JSON object"

    for query, entry in baseline.items():
        assert isinstance(entry, dict), f"entry for {query!r} must be a dict"
        assert "total" in entry, f"entry for {query!r} missing 'total'"
        assert "max" in entry, f"entry for {query!r} missing 'max'"
        assert "scores" in entry, f"entry for {query!r} missing 'scores'"
        assert 0 <= entry["total"] <= entry["max"]
        for dim in entry["scores"]:
            score = entry["scores"][dim]
            assert 0 <= score <= 2, f"score for {dim} out of range in {query!r}"


@pytest.mark.slow
def test_regression_suite() -> None:
    """Run the full regression suite against the current baseline.

    This test is marked ``slow`` — run it explicitly::

        pytest tests/test_regression.py -m slow -v

    or during CI, not on every commit.
    """
    if not _BASELINE_PATH.exists():
        # First run: just generate the baseline — don't fail.
        results = asyncio.run(_run_and_save_baseline())
        assert results, "first baseline generation failed"
        return

    baseline = load_baseline()
    if not baseline:
        pytest.skip("Baseline is empty")

    # Run queries in parallel batches of 2 to respect API rate limits
    results: list[dict] = []
    for query in TEST_QUERIES:
        result = asyncio.run(run_single(query))
        results.append(result)

    comparison = diff(results, baseline)

    if comparison["regressions"]:
        regressed_queries = list(comparison["regressions"].keys())
        pytest.fail(
            f"{len(regressed_queries)} regression(s) detected:\n" +
            "\n".join(f"  - {q}: {comparison['regressions'][q]['baseline_total']}"
                      f" → {comparison['regressions'][q]['new_total']}"
                      for q in regressed_queries)
        )


async def _run_and_save_baseline() -> list[dict]:
    from compintel.regression_test import save_baseline
    results = []
    for query in TEST_QUERIES:
        result = await run_single(query)
        results.append(result)
    baseline = {
        r["query"]: {"total": r["total"], "max": r["max"], "scores": r["scores"]}
        for r in results if "error" not in r
    }
    save_baseline(baseline)
    return results
