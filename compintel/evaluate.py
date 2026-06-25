"""Automated report quality evaluator for CompIntel Research.

Usage:
    python -m compintel.evaluate outputs/compintel_bundle_20260619_xxx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_bundle(bundle_path: Path) -> dict[str, Any]:
    """Load snapshot.json and report.md from a bundle directory."""
    snapshot_path = bundle_path / "snapshot.json"
    report_path = bundle_path / "report.md"
    if not snapshot_path.exists():
        sys.exit(f"snapshot.json not found in {bundle_path}")
    with open(snapshot_path, encoding="utf-8") as f:
        snapshot = json.load(f)
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {"result": snapshot.get("result", {}), "report": report}


def _safe_list(value: Any) -> list[Any]:
    """Coerce *value* to a list, handling None and scalar values safely."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def score_competitor_authenticity(result: dict[str, Any]) -> tuple[int, str]:
    """1. 竞品真实性 — all competitor names are real companies (0/1/2)."""
    profiles = result.get("profiles", [])
    competitors = result.get("competitors", [])
    names = [p.get("name", "") for p in profiles]
    names.extend(c.get("name", "") for c in competitors)
    names = [n for n in names if n.strip()]
    if not names:
        return 0, "无竞品数据"

    fake_count = sum(1 for n in names if "alternative" in n.lower() or "替代" in n)
    if fake_count > 0:
        return 0, f"发现 {fake_count} 个假竞品名（含 'Alternative'）"
    # Count how many have real source URLs
    sources = result.get("report", {}).get("sources", [])
    real_sources = [s for s in sources if s.startswith("http")]
    if len(real_sources) >= len(names):
        return 2, f"全部 {len(names)} 个竞品名来自真实搜索"
    return 1, f"{len(real_sources)}/{len(names)} 个竞品有真实来源 URL"


def score_market_segment_accuracy(result: dict[str, Any]) -> tuple[int, str]:
    """2. 赛道准确性 — competitors belong to the same market segment (0/1/2)."""
    market = result.get("report", {}).get("market_analysis", {})
    landscape = market.get("competitive_landscape", {})
    if not landscape:
        return 1, "无 competitive_landscape 数据，无法判断"

    all_players = (
        _safe_list(landscape.get("leaders"))
        + _safe_list(landscape.get("challengers"))
        + _safe_list(landscape.get("niche", landscape.get("niche_players")))
    )
    if not all_players:
        return 1, "competitive_landscape 为空"

    # Check if names that shouldn't be SaaS are mixed with SaaS seed data
    seed_keywords = [
        "notion", "coda", "airtable", "slack", "teams", "asana", "clickup",
        "confluence", "trello", "miro", "linear", "monday",
    ]
    cross_industry = [p for p in all_players if isinstance(p, str) and p.lower() in seed_keywords]
    if cross_industry:
        return 0, f"混入 SaaS 工具名：{cross_industry}，赛道判断错误"
    return 2, f"所有 {len(all_players)} 个竞品未被检测到赛道冲突"


def score_rag_memory_utilization(result: dict[str, Any]) -> tuple[int, str]:
    """3. RAG 记忆利用 — RAG retrieves historical analysis, not just seeds (0/1/2)."""
    profiles = result.get("profiles", [])
    all_ctx = []
    for p in profiles:
        all_ctx.extend(p.get("rag_context", []))

    if not all_ctx:
        return 0, "RAG 返回空"

    seed_count = sum(1 for c in all_ctx if c.get("metadata", {}).get("report_type") in ("seed", "seed_competitor_report", None))
    historical_count = sum(1 for c in all_ctx if c.get("metadata", {}).get("report_type") in ("swot", "market_analysis", "executive_summary"))
    if historical_count > 0:
        return 2, f"{historical_count}/{len(all_ctx)} 条来自历史分析（swot/market_analysis），{seed_count} 条是种子数据"
    if seed_count > 0:
        return 1, f"全部 {seed_count} 条来自种子数据，无历史分析记忆"
    return 0, "RAG 返回数据无有效 metadata"


def score_source_traceability(result: dict[str, Any]) -> tuple[int, str]:
    """4. Source 可追溯 — SWOT evidence is real text, not labels (0/1/2)."""
    swot = result.get("report", {}).get("swot_analysis", {})
    competitors = swot.get("competitors", [])

    evidence_items: list[str] = []
    for comp in competitors:
        for quadrant in ("strengths", "weaknesses", "opportunities", "threats"):
            for item in comp.get(quadrant, []):
                ev = (item.get("evidence") or "").strip()
                if ev:
                    evidence_items.append(ev)

    if not evidence_items:
        return 0, "无 SWOT evidence"

    label_names = {"tavily", "search_worker", "scrape_worker", "rag_retriever", "profile summary"}
    label_count = sum(1 for e in evidence_items if e.lower() in label_names)
    url_count = sum(1 for e in evidence_items if e.startswith("http"))
    real_text_count = len(evidence_items) - label_count - url_count

    if real_text_count >= len(evidence_items) * 0.5:
        return 2, f"{real_text_count}/{len(evidence_items)} 条 evidence 是真实文本"
    if url_count > label_count:
        return 1, f"{url_count} URLs, {label_count} 内部标签, {real_text_count} 真实文本"
    return 0, f"{label_count}/{len(evidence_items)} 条 evidence 是内部标签（tavily 等）"


def score_degradation_behavior(log_events: list[dict[str, Any]], report: str) -> tuple[int, str]:
    """5. 降级行为 — LLM failures handled gracefully with meaningful fallback (0/1/2)."""
    if not log_events:
        return 1, "无执行日志"

    # Count LLM failures in execution log
    error_events = [e for e in log_events if "error" in (e.get("event") or "").lower() or "failed" in (e.get("detail") or "").lower()]
    llm_fail_count = sum(1 for e in log_events if "LLM call failed" in (e.get("detail") or ""))

    # Check report quality signals
    placeholder_count = report.lower().count("placeholder")
    template_phrase = "should be evaluated against competitor product scope"
    has_template_conclusion = template_phrase in report.lower()
    has_data_conclusion = "search results" in report.lower() and "scraped pages" in report.lower()

    if llm_fail_count == 0 and placeholder_count == 0:
        return 2, "无 LLM 失败，零 placeholder，完整 LLM 质量"
    if has_data_conclusion and placeholder_count <= 3:
        return 2, f"{llm_fail_count} 次 LLM 失败但降级产出了数据驱动 conclusion，placeholder={placeholder_count}"
    if placeholder_count == 0 and not has_template_conclusion:
        return 1, f"{llm_fail_count} 次 LLM 失败，降级质量可接受"
    if has_template_conclusion:
        return 0, f"conclusion 为万能模板（'{template_phrase[:50]}...'）"
    return 0, f"placeholder={placeholder_count}，降级质量不可接受"


def score_pipeline_completeness(log_events: list[dict[str, Any]]) -> tuple[int, str]:
    """6. Pipeline 完整性 — all 8 nodes executed, rag_ingest wrote to disk (0/1/2)."""
    if not log_events:
        return 0, "无执行日志"

    node_names = {e.get("node") for e in log_events if e.get("node")}
    required = {"intent_analyst", "research_planner", "competitor_profiler",
                "market_analyst", "swot_synthesizer", "report_writer", "reviewer"}
    missing = required - node_names
    if missing:
        return 0, f"缺失节点: {missing}"

    rag_ingest = [e for e in log_events if e.get("node") == "rag_ingest"]
    if not rag_ingest:
        return 1, "7 主节点全过，但 rag_ingest 未执行"
    ingest_event = rag_ingest[-1]
    if "ingested" in (ingest_event.get("detail") or "").lower():
        return 2, f"8/8 节点全过，rag_ingest: {ingest_event.get('detail')}"
    return 1, f"rag_ingest 执行但未写入数据"


def score_report_readability(result: dict[str, Any], report: str) -> tuple[int, str]:
    """7. 报告可读性 — structured, non-template, independently readable (0/1/2)."""
    report_data = result.get("report", {}).get("report", {})
    exec_summary = report_data.get("executive_summary", "")
    sections = report_data.get("sections", [])
    conclusion = report_data.get("conclusion", "")
    profiles = result.get("profiles", [])

    issues: list[str] = []

    # 1. Placeholder count
    placeholder_count = report.lower().count("placeholder")
    if placeholder_count >= 3:
        issues.append(f"全文出现 {placeholder_count} 处 'placeholder'")
    elif placeholder_count > 0:
        issues.append(f"全文出现 {placeholder_count} 处 'placeholder'")

    # 2. Executive summary quality
    template_patterns = [
        "competitive analysis based on collected profiles",
        "analysis for ",
    ]
    if any(p in exec_summary.lower() for p in template_patterns) and len(exec_summary) < 100:
        issues.append("Executive Summary 是模板填充")

    # 3. Profile summary quality — "Profile summary for X" is template
    template_profile_count = sum(
        1 for p in profiles
        if (p.get("summary") or "").startswith("Profile summary for")
    )
    if template_profile_count == len(profiles) > 0:
        issues.append(f"全部 {template_profile_count} 个竞品 summary 为模板 'Profile summary for X'")
    elif template_profile_count > 0:
        issues.append(f"{template_profile_count}/{len(profiles)} 个竞品 summary 为模板")

    # 4. Structure completeness
    required_sections = {"Competitor Profiles", "Market Analysis", "SWOT"}
    section_titles = {s.get("title", "") for s in sections}
    missing_sections = required_sections - section_titles
    if missing_sections:
        issues.append(f"缺失章节: {missing_sections}")

    # 5. Data gaps present
    data_gaps = report_data.get("data_gaps", [])
    if not data_gaps:
        issues.append("无 data_gaps 标注（可能是 LLM 未发现缺口，也可能是完美数据）")

    if not issues:
        return 2, "报告结构完整，无模板填充，可独立阅读"
    if len(issues) <= 2 and placeholder_count == 0:
        return 1, f"可接受：{'；'.join(issues)}"
    return 0, f"不可接受：{'；'.join(issues)}"


def evaluate(bundle_path: Path) -> dict[str, Any]:
    """Run all 7 scoring dimensions against a completed analysis bundle."""
    bundle = load_bundle(bundle_path)
    result = bundle["result"]
    report = bundle["report"]
    log_events = result.get("report", {}).get("execution_log", [])

    dimensions = {
        "1.竞品真实性": score_competitor_authenticity(result),
        "2.赛道准确性": score_market_segment_accuracy(result),
        "3.RAG记忆利用": score_rag_memory_utilization(result),
        "4.Source可追溯": score_source_traceability(result),
        "5.降级行为": score_degradation_behavior(log_events, report),
        "6.Pipeline完整性": score_pipeline_completeness(log_events),
        "7.报告可读性": score_report_readability(result, report),
    }

    total = sum(score for score, _ in dimensions.values())
    return {"scores": dimensions, "total": total, "max": 14, "bundle": str(bundle_path)}


def format_card(eval_result: dict[str, Any]) -> str:
    """Format evaluation result as a terminal-friendly scorecard."""
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append("  CompIntel Report Quality Evaluation")
    lines.append("=" * 64)
    lines.append(f"  Bundle: {Path(eval_result['bundle']).name}")
    lines.append("")
    lines.append(f"  {'#':<20} {'Score':>5}  Evidence")
    lines.append(f"  {'-'*18:<20} {'-'*3:>5}  {'-'*36}")
    for dim, (score, evidence) in eval_result["scores"].items():
        bar = "##" if score == 2 else "#-" if score == 1 else "--"
        lines.append(f"  {dim:<20} {bar} {score}/2  {evidence}")
    lines.append(f"  {'-'*64}")
    total = eval_result["total"]
    max_s = eval_result["max"]
    pct = total / max_s * 100
    grade = "A" if total >= 12 else "B" if total >= 9 else "C" if total >= 6 else "D"
    lines.append(f"  TOTAL: {total}/{max_s} ({pct:.0f}%) -- Grade: {grade}")
    lines.append("=" * 64)
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find the latest bundle
        outputs = Path("outputs")
        if outputs.exists():
            bundles = sorted(
                [d for d in outputs.iterdir() if d.is_dir() and d.name.startswith("compintel_bundle")],
                key=lambda d: d.stat().st_mtime, reverse=True,
            )
            if bundles:
                ev = evaluate(bundles[0])
                text = format_card(ev)
                try:
                    print(text)
                except UnicodeEncodeError:
                    print(text.encode("ascii", errors="replace").decode("ascii"))
                sys.exit(0)
        sys.exit("Usage: python -m compintel.evaluate <bundle_path>")

    bundle_path = Path(sys.argv[1])
    if not bundle_path.exists():
        sys.exit(f"Bundle not found: {bundle_path}")
    ev = evaluate(bundle_path)
    text = format_card(ev)
    try:
        print(text)
    except UnicodeEncodeError:
        # Windows GBK terminal fallback
        print(text.encode("ascii", errors="replace").decode("ascii"))
