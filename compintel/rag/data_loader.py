"""Seed data loading for CompIntel RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .base import RagDocument
from .qdrant_store import QdrantStore


DEFAULT_SEED_REPORTS: dict[str, str] = {
    # SaaS / collaboration
    "Notion": "Notion is an all-in-one workspace platform combining docs, wikis, and project management. Competes with Coda, Confluence, and Microsoft Loop in the collaborative knowledge management market.",
    "Coda": "Coda is a collaborative document platform that combines spreadsheets, databases, and workflow automation. Competes with Notion, Airtable, and Google Workspace.",
    "Airtable": "Airtable is a low-code platform for building relational databases and collaborative apps. Competes with Notion, Coda, and Monday.com in the flexible work-management space.",
    "Slack": "Slack is a team communication platform with channels, integrations, and workflow automation. Competes with Microsoft Teams, Discord, and Google Chat in enterprise messaging.",
    "Microsoft Teams": "Microsoft Teams is an enterprise collaboration hub integrating chat, video, and Office 365. Competes with Slack, Zoom, and Google Workspace.",
    "Monday.com": "Monday.com is a work operating system for project management and team collaboration. Competes with Asana, ClickUp, and Smartsheet.",
    "ClickUp": "ClickUp is an all-in-one productivity platform with tasks, docs, goals, and time tracking. Competes with Monday.com, Asana, and Notion in project management.",
    "Confluence": "Confluence by Atlassian is a team workspace for documentation, knowledge bases, and project collaboration. Competes with Notion, GitBook, and Slab.",
    "Trello": "Trello by Atlassian is a visual kanban-based project management tool. Competes with Asana, Monday.com, and Linear in lightweight task management.",
    "Miro": "Miro is a visual collaboration platform for whiteboarding, diagramming, and workshops. Competes with FigJam, Lucid, and Mural in visual ideation tools.",
    "Linear": "Linear is a project management platform designed for software product teams. Competes with Jira, Shortcut, and Height in developer-centric project tracking.",

    # VC / investment
    "Sequoia Capital": "Sequoia Capital is a leading venture capital firm investing in seed, early, and growth-stage technology companies globally. Key portfolio includes Stripe, WhatsApp, and DoorDash.",
    "Andreessen Horowitz": "Andreessen Horowitz (a16z) is a venture capital firm investing across AI, crypto, fintech, bio, and enterprise. Known for portfolio services and thesis-driven investing.",
    "Tiger Global Management": "Tiger Global is an investment firm focused on public and private technology companies with a high-velocity, low-touch approach. Competes with SoftBank Vision Fund in growth-stage tech.",
    "SoftBank Vision Fund": "SoftBank Vision Fund is the world's largest technology investment fund, deploying capital across AI, robotics, fintech, and telecom. Competes with sovereign wealth funds and crossover investors.",
    "Accel": "Accel is a global venture capital firm investing in early and growth-stage technology companies. Key exits include Facebook, Slack, and Atlassian.",
    "Benchmark": "Benchmark is an early-stage venture capital firm known for concentrated investments and operator-focused partnership. Portfolio includes Uber, Twitter, and Docker.",

    # Big tech / AI
    "Google": "Google (Alphabet) is a technology conglomerate with leading positions in search, advertising, cloud computing, and AI research. Competes with Microsoft and Amazon across multiple verticals.",
    "Microsoft": "Microsoft is a global technology company with dominant positions in enterprise software, cloud computing (Azure), and AI (via OpenAI partnership). Competes with Google, Amazon, and Apple.",
    "Amazon": "Amazon is a global e-commerce and cloud computing leader (AWS) expanding into AI, logistics, and healthcare. Competes with Google Cloud, Microsoft Azure, and Walmart.",
    "Meta": "Meta (formerly Facebook) is a social media and technology company investing heavily in AI, VR/AR, and the metaverse. Competes with TikTok, Google, and Apple in digital advertising.",
    "Apple": "Apple is a consumer technology company with iPhone, Mac, and services ecosystem. Expanding into AI via Apple Intelligence and spatial computing via Vision Pro.",
    "NVIDIA": "NVIDIA is the leading AI computing company, dominating the GPU market for training and inference. Powers most large language model infrastructure.",
    "OpenAI": "OpenAI is the leading AI research company behind ChatGPT, GPT-4, and DALL-E. Competes with Anthropic, Google DeepMind, and Meta AI in frontier AI models.",
    "Anthropic": "Anthropic is an AI safety-focused research company developing Claude (large language model). Competes with OpenAI, Google DeepMind, and Cohere.",
    "DeepSeek": "DeepSeek is a Chinese AI research lab producing competitive open-weight and proprietary LLMs. Competes with OpenAI, Anthropic, and Qwen in AI model development.",

    # Consumer / hardware
    "Tesla": "Tesla is an electric vehicle and clean energy company. Competes with BYD, Rivian, and traditional automakers in the global EV transition.",
    "BYD": "BYD is a Chinese electric vehicle and battery manufacturer, the world's largest NEV maker by volume. Competes with Tesla, NIO, and Volkswagen in the global EV market.",
    "Samsung": "Samsung is a global electronics conglomerate with leadership in smartphones, semiconductors, and displays. Competes with Apple, TSMC, and LG across consumer and industrial segments.",
    "Sony": "Sony is a Japanese electronics and entertainment conglomerate spanning gaming, music, film, and imaging sensors. Competes with Microsoft (Xbox), Nintendo, and Samsung.",
}


_SEED_COMPETITOR_DEFAULTS = list(DEFAULT_SEED_REPORTS.keys())


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
                "competitor": name,
                "source": f"seed:{name}",
                "text": summary,
            }
            for name, summary in DEFAULT_SEED_REPORTS.items()
        ]


def load_seed_reports(store: QdrantStore | None = None, reports: Iterable[dict[str, Any]] | None = None) -> int:
    return SeedReportLoader(store=store or QdrantStore()).load_seed_reports(reports)
