"""Markdown report formatting for CompIntel Research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MarkdownFormatter:
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, payload: dict[str, Any]) -> str:
        result = payload.get("result") or {}
        intent = result.get("intent") or {}
        profiles = result.get("profiles") or []
        report_container = result.get("report") or {}
        report = report_container.get("report") or report_container
        market_analysis = report.get("market_analysis") or report_container.get("market_analysis", {})
        swot_analysis = report.get("swot_analysis") or report_container.get("swot_analysis", {})
        review_feedback = report.get("review_feedback") or report_container.get("review_feedback", {})
        sources = self._dedupe_sources(report.get("sources", []))
        if not sources:
            sources = self._collect_profile_sources(profiles)

        lines = [
            f"# CompIntel Research Report: {intent.get('target', 'unknown')}",
            "",
            "## Executive Summary",
            str(report.get("executive_summary", "") or "No executive summary available."),
            "",
            "## Market Segment",
            str(intent.get("market_segment", "unknown")),
            "",
            "## Competitors",
        ]
        for competitor in result.get("competitors", []):
            lines.append(f"- {competitor.get('name', 'unknown')}")
        lines.extend(
            [
                "",
                "## Competitor Comparison",
                "| Name | Summary | Sources |",
                "|---|---|---:|",
            ]
        )
        for profile in profiles:
            lines.append(
                "| {name} | {summary} | {source_count} |".format(
                    name=self._escape_table(profile.get("name", "unknown")),
                    summary=self._escape_table(profile.get("summary", "")),
                    source_count=len(self._collect_profile_sources([profile])),
                )
            )
        if not profiles:
            lines.append("| unknown | No profile data available. | 0 |")

        if report.get("sections"):
            lines.extend(["", "## Report Sections"])
            for section in report.get("sections", []):
                if isinstance(section, dict):
                    lines.extend(
                        [
                            "",
                            f"### {section.get('title', 'Untitled')}",
                            str(section.get("content", "")),
                        ]
                    )
                    insights = section.get("key_insights") or []
                    if insights:
                        lines.append("")
                        lines.append("Key insights:")
                        for insight in insights:
                            lines.append(f"- {insight}")
                elif str(section).strip():
                    lines.append(f"- {section}")

        lines.extend(
            [
                "",
                "## Market Analysis",
                self._render_block(market_analysis),
                "",
                "## SWOT",
                self._render_swot(swot_analysis),
                "",
                "## Data Gaps",
                self._render_data_gaps(report.get("data_gaps", [])),
                "",
                "## Sources",
                self._render_sources(sources),
                "",
                "## Review",
                self._render_block(review_feedback),
                "",
            ]
        )
        if report.get("conclusion"):
            lines.extend(["## Conclusion", str(report.get("conclusion")), ""])
        return "\n".join(lines)

    def save(self, payload: dict[str, Any], filename: str = "compintel_report.md") -> Path:
        path = self.output_dir / filename
        path.write_text(self.render(payload), encoding="utf-8")
        return path

    def _render_block(self, value: Any) -> str:
        if isinstance(value, dict):
            return "\n".join(f"- {key}: {item}" for key, item in value.items()) or "- empty"
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value) or "- empty"
        return str(value)

    def _render_swot(self, value: Any) -> str:
        if not isinstance(value, dict):
            return self._render_block(value)

        lines: list[str] = []
        if value.get("summary"):
            lines.extend([str(value["summary"]), ""])

        competitors = value.get("competitors", [])
        if isinstance(competitors, list):
            for competitor in competitors:
                if not isinstance(competitor, dict):
                    continue
                lines.append(f"### {competitor.get('name', 'unknown')}")
                for key, label in (
                    ("strengths", "Strengths"),
                    ("weaknesses", "Weaknesses"),
                    ("opportunities", "Opportunities"),
                    ("threats", "Threats"),
                ):
                    lines.append(f"#### {label}")
                    lines.extend(self._render_swot_items(competitor.get(key, [])))
                    lines.append("")

        cross = value.get("cross_analysis")
        if isinstance(cross, dict):
            lines.append("### Cross Analysis")
            for key, label in (
                ("common_strengths", "Common Strengths"),
                ("differentiators", "Differentiators"),
            ):
                lines.append(f"#### {label}")
                lines.extend(self._render_swot_items(cross.get(key, [])))
                lines.append("")

        return "\n".join(lines).strip() or "- empty"

    def _render_swot_items(self, items: Any) -> list[str]:
        lines: list[str] = []
        values = items if isinstance(items, list) else [items]
        for item in values:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                evidence = str(item.get("evidence", "")).strip()
                if text:
                    lines.append(f"- {text}")
                if evidence:
                    lines.append(f"  - Evidence: {evidence}")
            elif str(item).strip():
                lines.append(f"- {item}")
        return lines or ["- empty"]

    def _render_data_gaps(self, gaps: Any) -> str:
        values = gaps if isinstance(gaps, list) else [gaps]
        lines = [f"- ⚠ Data Gap: {gap}" for gap in values if str(gap).strip()]
        return "\n".join(lines) or "- No explicit data gaps."

    def _render_sources(self, sources: list[str]) -> str:
        if not sources:
            return "- No traceable sources."
        lines = []
        for idx, source in enumerate(sources, start=1):
            if source.startswith("http://") or source.startswith("https://"):
                lines.append(f"{idx}. [{source}]({source})")
            else:
                lines.append(f"{idx}. {source}")
        return "\n".join(lines)

    def _dedupe_sources(self, values: Any) -> list[str]:
        sources: list[str] = []
        for value in values if isinstance(values, list) else [values]:
            source = ""
            if isinstance(value, dict):
                source = str(value.get("url") or value.get("source") or value.get("title") or "")
            else:
                source = str(value)
            source = source.strip()
            if source in {"search_worker", "scrape_worker", "rag_retriever"}:
                continue
            if source and source not in sources:
                sources.append(source)
        return sources

    def _collect_profile_sources(self, profiles: list[dict[str, Any]]) -> list[str]:
        sources: list[str] = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            for key in ("search_results", "scraped_content", "rag_context", "sources"):
                for source in self._dedupe_sources(profile.get(key, [])):
                    if source not in sources:
                        sources.append(source)
        return sources

    def _escape_table(self, value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").strip()
