"""Seed data loading for CompIntel RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .base import RagDocument
from .qdrant_store import QdrantStore


DEFAULT_SAAS_COMPETITORS = [
    "Notion",
    "Coda",
    "Airtable",
    "Slack",
    "Microsoft Teams",
    "Asana",
    "Monday.com",
    "ClickUp",
    "Confluence",
    "Trello",
    "Miro",
    "Linear",
]


@dataclass(slots=True)
class SeedReportLoader:
    store: QdrantStore = field(default_factory=QdrantStore)

    def load_seed_reports(self, reports: Iterable[dict[str, Any]] | None = None) -> int:
        documents = [
            self._report_to_document(report)
            for report in (reports if reports is not None else self._default_reports())
        ]
        return self.store.ingest(documents)

    def _report_to_document(self, report: dict[str, Any]) -> RagDocument:
        competitor = str(report.get("competitor", "unknown"))
        text = str(report.get("text", ""))
        return RagDocument(
            text=text,
            source=str(report.get("source", f"seed:{competitor}")),
            metadata={
                "competitor": competitor,
                "report_type": report.get("report_type", "seed_competitor_report"),
            },
        )

    def _default_reports(self) -> list[dict[str, Any]]:
        return [
            {
                "competitor": competitor,
                "source": f"seed:{competitor}",
                "text": (
                    f"{competitor} is a SaaS competitor tracked by CompIntel Research. "
                    f"Seed notes should be replaced by GPT Researcher generated reports "
                    f"before production RAG evaluation."
                ),
            }
            for competitor in DEFAULT_SAAS_COMPETITORS
        ]


def load_seed_reports(store: QdrantStore | None = None, reports: Iterable[dict[str, Any]] | None = None) -> int:
    return SeedReportLoader(store=store or QdrantStore()).load_seed_reports(reports)
