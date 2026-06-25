"""Automated regression evaluator for CompIntel Research.

Usage::

    # Run all test queries and compare against baseline
    python -m compintel.regression_test

    # Run and update baseline (after verifying all scores improved or held)
    python -m compintel.regression_test --update-baseline

    # Run a single query for debugging
    python -m compintel.regression_test --query "分析 Notion 的竞品格局"

The baseline file lives at ``outputs/regression_baseline.json`` and is
updated by the user *after* reviewing the diff.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .bundle import generate_delivery_bundle
from .evaluate import evaluate
from .execution import CompIntelExecution

# ── Test query bank ────────────────────────────────────────────────────
#
# Covers the key usage paths and known fragile points:
#   1. Chinese single-competitor (most common real-world use)
#   2. English query (cross-language path)
#   3. Multi-competitor comparison (fan-out stress)
#   4. No explicit competitor — relies on seed discovery
#   5. Investment perspective (different analysis angle)
#   6. Niche market (few scrape sources, tests degradation)
#   7. Vague query (stress-tests intent parsing)
TEST_QUERIES: list[str] = [
    "分析 Notion 的竞品格局",
    "Analyze Tesla competitors in EV market",
    "对比 比亚迪 特斯拉 蔚来 小鹏 在电动汽车市场的竞争策略",
    "分析 Figma 的竞品格局",
    "分析小米在智能汽车领域的投资布局和竞争策略",
    "分析瑞幸咖啡的竞品格局",
    "知识管理工具有哪些主要玩家及其竞争对比分析",
    "分析 Slack 在团队协作市场的竞品格局",
    "Compare BYD NIO in global EV market competitive landscape",
    "分析 Zoom 在视频会议市场的竞争态势",
]

_BASELINE_PATH = Path("outputs/regression_baseline.json")


# ── public API ─────────────────────────────────────────────────────────


def load_baseline() -> dict[str, Any]:
    """Load the saved baseline scores or return an empty dict."""
    if _BASELINE_PATH.exists():
        return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    return {}


def save_baseline(scores: dict[str, Any]) -> None:
    _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_PATH.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


async def run_single(query: str) -> dict[str, Any]:
    """Run the full pipeline for one query and return the evaluation result."""
    execution = CompIntelExecution()
    exec_result = await execution.run_intent(query)

    bundle = generate_delivery_bundle(exec_result, output_dir="outputs")
    eval_result = evaluate(Path(bundle["bundle_path"]))

    return {
        "query": query,
        "total": eval_result["total"],
        "max": eval_result["max"],
        "scores": {
            dimension: score
            for dimension, (score, _evidence) in eval_result["scores"].items()
        },
        "evidence": {
            dimension: evidence
            for dimension, (_score, evidence) in eval_result["scores"].items()
        },
        "bundle": bundle["bundle_path"],
    }


async def run_all() -> list[dict[str, Any]]:
    """Run all test queries sequentially and return evaluation results."""
    results: list[dict[str, Any]] = []
    total = len(TEST_QUERIES)

    print(f"\nRunning {total} regression queries...\n")

    for idx, query in enumerate(TEST_QUERIES, 1):
        sys.stdout.write(f"  [{idx}/{total}] {query[:60]}...")
        sys.stdout.flush()
        try:
            result = await run_single(query)
            pct = result["total"] / result["max"] * 100
            grade_char = (
                "A" if result["total"] >= 12
                else "B" if result["total"] >= 9
                else "C" if result["total"] >= 6
                else "D"
            )
            sys.stdout.write(f" {result['total']}/{result['max']} ({pct:.0f}%) {grade_char}\n")
            results.append(result)
        except Exception as exc:
            sys.stdout.write(f" ERROR: {exc}\n")
            results.append({"query": query, "total": 0, "max": 14, "scores": {}, "error": str(exc)})
            continue

    return results


def diff(
    results: list[dict[str, Any]],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare *results* against *baseline* and flag regressions.

    Returns a dict of {query: {dimension: delta}} where delta < 0 means
    the score *dropped* (a regression).  Queries that are new (not in
    baseline) are reported separately.
    """
    if baseline is None:
        baseline = load_baseline()

    comparison: dict[str, Any] = {
        "regressions": {},
        "improvements": {},
        "unchanged": {},
        "new_queries": {},
        "summary": {"total_queries": len(results),
                    "regressed": 0, "improved": 0, "unchanged": 0, "new": 0},
    }

    for entry in results:
        query = entry["query"]
        if query not in baseline:
            comparison["new_queries"][query] = entry
            comparison["summary"]["new"] += 1
            continue

        base = baseline[query]
        base_total = base.get("total", 0)
        new_total = entry.get("total", 0)
        delta_total = new_total - base_total

        dims: dict[str, int] = {}
        for dim in base.get("scores", {}):
            old_s = base["scores"].get(dim, 0)
            new_s = entry.get("scores", {}).get(dim, 0)
            if new_s != old_s:
                dims[dim] = new_s - old_s

        if delta_total < 0:
            comparison["regressions"][query] = {
                "baseline_total": base_total,
                "new_total": new_total,
                "delta_total": delta_total,
                "deltas": dims,
                "bundle": entry.get("bundle", ""),
            }
            comparison["summary"]["regressed"] += 1
        elif delta_total > 0:
            comparison["improvements"][query] = {
                "baseline_total": base_total,
                "new_total": new_total,
                "delta_total": delta_total,
                "deltas": dims,
                "bundle": entry.get("bundle", ""),
            }
            comparison["summary"]["improved"] += 1
        else:
            comparison["unchanged"][query] = new_total
            comparison["summary"]["unchanged"] += 1

    return comparison


# ── formatting ─────────────────────────────────────────────────────────


def format_diff(comparison: dict[str, Any]) -> str:
    """Render the regression diff as a terminal-friendly table."""
    lines: list[str] = []
    s = comparison["summary"]

    lines.append("")
    lines.append("=" * 72)
    lines.append("  CompIntel Regression Test Results")
    lines.append("=" * 72)
    lines.append(
        f"  {s['total_queries']} queries: "
        f"{s['regressed']} regressed, "
        f"{s['improved']} improved, "
        f"{s['unchanged']} unchanged, "
        f"{s['new']} new"
    )
    lines.append("")

    # ── Regressions (most important, shown first) ──
    if comparison["regressions"]:
        lines.append("  REGRESSIONS (score dropped):")
        lines.append(f"  {'Query':<42} {'Old':>5} {'New':>5} {'Δ':>5}")
        lines.append(f"  {'-'*40:<42} {'-'*3:>5} {'-'*3:>5} {'-'*3:>5}")
        for query, info in sorted(comparison["regressions"].items(),
                                   key=lambda kv: kv[1]["delta_total"]):
            lines.append(
                f"  {query[:40]:<42} "
                f"{info['baseline_total']:>3}/{14:>3} "
                f"{info['new_total']:>3}/{14:>3} "
                f"{info['delta_total']:>+4}"
            )
            for dim, delta in sorted(info.get("deltas", {}).items()):
                lines.append(f"    └─ {dim}: {delta:+d}")
        lines.append("")

    # ── Improvements ──
    if comparison["improvements"]:
        lines.append("  IMPROVEMENTS:")
        for query, info in sorted(comparison["improvements"].items(),
                                   key=lambda kv: -kv[1]["delta_total"]):
            lines.append(
                f"    {query[:50]}  "
                f"{info['baseline_total']}/{14} → {info['new_total']}/{14} "
                f"(+{info['delta_total']})"
            )
        lines.append("")

    # ── New queries ──
    if comparison["new_queries"]:
        lines.append("  NEW (not in baseline):")
        for query, entry in comparison["new_queries"].items():
            lines.append(f"    {query[:50]}  {entry['total']}/{entry['max']}")
        lines.append("")

    # ── Verdict ──
    if comparison["regressions"]:
        lines.append(f"  VERDICT: {s['regressed']} regression(s) detected — fix before committing.")
    elif comparison["new_queries"]:
        lines.append(f"  VERDICT: All {s['unchanged'] + s['improved']} existing queries stable."
                     f"  {s['new']} new query(s) — run with --update-baseline to capture.")
    else:
        lines.append(f"  VERDICT: All {s['unchanged'] + s['improved']} queries stable.  No regressions.")

    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="CompIntel regression test")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Save current scores as the new baseline after running.")
    parser.add_argument("--query", type=str,
                        help="Run a single query instead of the full test suite.")
    args = parser.parse_args()

    if args.query:
        results = [asyncio.run(run_single(args.query))]
    else:
        results = asyncio.run(run_all())

    # Always diff against the existing baseline
    baseline = load_baseline()
    comparison = diff(results, baseline)
    print(format_diff(comparison))

    if args.update_baseline:
        if comparison["regressions"] and not args.query:
            print("ERROR: Cannot update baseline — there are regressions.")
            print("Fix the regressions first, or use --query to test a single query.")
            sys.exit(1)
        new_baseline = {
            entry["query"]: {
                "total": entry["total"],
                "max": entry["max"],
                "scores": entry.get("scores", {}),
            }
            for entry in results
            if "error" not in entry
        }
        # Merge with existing baseline (preserve entries not in this run)
        merged = {**baseline, **new_baseline} if not args.query else {**baseline, **new_baseline}
        save_baseline(merged)
        print(f"Baseline updated: {_BASELINE_PATH} ({len(merged)} entries)")

    if comparison["regressions"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
