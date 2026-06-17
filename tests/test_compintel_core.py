from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from compintel.execution import CompIntelExecution
from compintel.bundle import BundleWriter, generate_delivery_bundle
from compintel.export import MarkdownFormatter
from compintel.progress import ProgressSummaryFormatter
from compintel.rag import QdrantStore, RagDocument, SeedReportLoader
from compintel.settings import CompIntelSettings
from compintel.agents.scrape_worker import ScrapeWorker
from compintel.agents.search_worker import SearchWorker
from compintel.tracker import ExecutionTracker


def test_tracker_snapshot_and_audit() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        audit_path = Path(temp_dir) / "audit.jsonl"
        tracker = ExecutionTracker(objective="CompIntel", audit_store=None)
        tracker.add_checkpoint("intent_analyst", "running", owner="team", summary="start", evidence=["q"])
        tracker.record_decision("test decision")
        snapshot = tracker.snapshot()

        assert snapshot.objective == "CompIntel"
        assert snapshot.checkpoints[0].phase == "intent_analyst"
        assert snapshot.decisions == ["test decision"]
        assert snapshot.status == "in_progress"
        assert not audit_path.exists()


def test_markdown_formatter_writes_report() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        formatter = MarkdownFormatter(output_dir=Path(temp_dir))
        payload = {
            "result": {
                "intent": {"target": "Notion", "market_segment": "collaboration software"},
                "competitors": [{"name": "Coda"}],
                "profiles": [{"name": "Coda", "summary": "Docs + tables"}],
                "report": {
                    "market_analysis": {"market_overview": "overview"},
                    "swot_analysis": {"summary": "swot"},
                    "review_feedback": {"approved": True},
                },
            }
        }

        path = formatter.save(payload, "report.md")

        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "CompIntel Research Report: Notion" in text
        assert "- Coda" in text


def test_execution_emits_ordered_events() -> None:
    async def _run() -> dict:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["COMPINTEL_AUDIT_PATH"] = str(Path(temp_dir) / "audit.jsonl")
            execution = CompIntelExecution()
            return await execution.run_intent("分析 Notion 在协作工具市场的竞品")

    outcome = asyncio.run(_run())
    event_types = [event["type"] for event in outcome["events"]]

    assert event_types[0] == "execution_started"
    assert event_types[-1] == "execution_completed"
    assert "analysis_ready" in event_types
    assert outcome["tracker"]["status"] == "in_progress"


def test_progress_summary_writes_progress_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        formatter = ProgressSummaryFormatter(output_dir=Path(temp_dir))
        tracker = {
            "objective": "CompIntel Research",
            "current_phase": "week_4",
            "status": "in_progress",
            "checkpoints": [
                {
                    "phase": "intent_analyst",
                    "status": "completed",
                    "owner": "team",
                    "summary": "done",
                }
            ],
            "pending_questions": [],
            "decisions": ["run analysis"],
            "risks": [],
        }
        path = formatter.save(tracker, [{"type": "execution_completed", "message": "done"}], "progress.md")

        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "CompIntel Progress Summary" in text
        assert "execution_completed" in text


def test_bundle_writer_creates_bundle() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        payload = {
            "result": {
                "intent": {"target": "Notion", "market_segment": "collaboration software"},
                "competitors": [{"name": "Coda"}],
                "profiles": [{"name": "Coda", "summary": "Docs + tables"}],
                "report": {
                    "market_analysis": {"market_overview": "overview"},
                    "swot_analysis": {"summary": "swot"},
                    "review_feedback": {"approved": True},
                },
            },
            "tracker": {
                "objective": "CompIntel Research",
                "current_phase": "week_4",
                "status": "in_progress",
                "checkpoints": [],
                "pending_questions": [],
                "decisions": [],
                "risks": [],
            },
            "events": [{"type": "execution_completed", "message": "done"}],
            "audit_path": str(Path(temp_dir) / "audit.jsonl"),
        }

        bundle_dir = BundleWriter(output_dir=Path(temp_dir)).write(payload, "bundle")

        assert bundle_dir.exists()
        assert (bundle_dir / "report.md").exists()
        assert (bundle_dir / "progress.md").exists()
        assert (bundle_dir / "snapshot.json").exists()
        assert (bundle_dir / "manifest.txt").exists()


def test_bundle_writer_defaults_to_unique_name() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        payload = {
            "result": {"intent": {}, "competitors": [], "profiles": [], "report": {}},
            "tracker": {
                "objective": "CompIntel Research",
                "current_phase": "week_4",
                "status": "in_progress",
                "checkpoints": [],
                "pending_questions": [],
                "decisions": [],
                "risks": [],
            },
            "events": [],
            "audit_path": str(Path(temp_dir) / "audit.jsonl"),
        }

        writer = BundleWriter(output_dir=Path(temp_dir))
        first = writer.write(payload)
        second = writer.write(payload)

        assert first != second
        assert first.exists()
        assert second.exists()


def test_generate_delivery_bundle_returns_paths() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        payload = {
            "result": {"intent": {}, "competitors": [], "profiles": [], "report": {}},
            "tracker": {
                "objective": "CompIntel Research",
                "current_phase": "week_4",
                "status": "in_progress",
                "checkpoints": [],
                "pending_questions": [],
                "decisions": [],
                "risks": [],
            },
            "events": [],
            "audit_path": str(Path(temp_dir) / "audit.jsonl"),
        }

        paths = generate_delivery_bundle(payload, output_dir=Path(temp_dir), bundle_name="bundle")

        assert paths["bundle_path"].endswith("bundle")
        assert Path(paths["report_path"]).exists()
        assert Path(paths["progress_path"]).exists()
        assert Path(paths["snapshot_path"]).exists()
        assert Path(paths["manifest_path"]).exists()


def test_settings_supports_generic_provider_fields(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("LLM_API_KEY", "kimi-real-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.moonshot.cn/v1")
    monkeypatch.setenv("FAST_LLM", "moonshot-v1-8k")
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("SERPAPI_API_KEY", "tvly-real-key")

    settings = CompIntelSettings.from_env()

    assert settings.llm_provider == "kimi"
    assert settings.fast_llm == "openai:moonshot-v1-8k"
    assert settings.openai_api_key == "kimi-real-key"
    assert settings.openai_base_url == "https://api.moonshot.cn/v1"
    assert settings.search_provider == "tavily"
    assert settings.search_api_key == "tvly-real-key"


def test_settings_ignores_placeholder_secrets(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "replace-with-your-deepseek-api-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "tvly-your_tavily_key_here")

    settings = CompIntelSettings.from_env()

    assert settings.llm_api_key is None
    assert settings.search_api_key is None


def test_search_worker_uses_provider_client_and_dedupes_results() -> None:
    class FakeTavilyClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, **kwargs) -> dict:
            self.queries.append(query)
            results = [
                {
                    "title": f"Result {idx}",
                    "url": f"https://example.com/shared-{idx % 2}",
                    "content": f"Snippet {idx}",
                }
                for idx in range(14)
            ]
            return {"results": results}

    client = FakeTavilyClient()
    settings = CompIntelSettings(
        search_provider="tavily",
        search_api_key="tvly-real-key",
    )
    worker = SearchWorker(client=client, settings=settings)

    result = asyncio.run(
        worker(
            {
                "competitor": {"name": "Notion"},
                "research_questions": ["pricing", "enterprise strategy"],
            }
        )
    )

    assert client.queries == [
        "Notion pricing competitive analysis",
        "Notion enterprise strategy competitive analysis",
    ]
    assert len(result["search_results"]) == 2
    assert {item["source"] for item in result["search_results"]} == {"tavily"}


def test_search_worker_limits_results_to_twenty() -> None:
    class FakeTavilyClient:
        def search(self, query: str, **kwargs) -> dict:
            return {
                "results": [
                    {
                        "title": f"Result {idx}",
                        "url": f"https://example.com/{query.replace(' ', '-')}/{idx}",
                        "content": f"Snippet {idx}",
                    }
                    for idx in range(5)
                ]
            }

    settings = CompIntelSettings(search_provider="tavily", search_api_key="tvly-real-key")
    worker = SearchWorker(client=FakeTavilyClient(), settings=settings)

    result = asyncio.run(
        worker(
            {
                "competitor": {"name": "Slack"},
                "research_questions": [f"question {idx}" for idx in range(10)],
            }
        )
    )

    assert len(result["search_results"]) == 20


def test_search_worker_returns_error_without_api_key() -> None:
    settings = CompIntelSettings(search_provider="tavily", search_api_key=None)
    worker = SearchWorker(settings=settings)

    result = asyncio.run(
        worker({"competitor": {"name": "Teams"}, "research_questions": ["pricing"]})
    )

    assert result["search_results"][0]["error"] is True
    assert "not configured" in result["search_results"][0]["message"]


def test_scrape_worker_builds_targets_and_truncates_content() -> None:
    class FakeScraper:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def scrape(self, url: str, user_agent: str) -> dict:
            self.urls.append(url)
            return {"url": url, "title": "Example", "content": "x" * 25}

    scraper = FakeScraper()
    worker = ScrapeWorker(
        scraper=scraper,
        max_chars=10,
        min_delay=0,
        max_delay=0,
    )

    result = asyncio.run(
        worker({"competitor": {"name": "Notion", "website": "notion.so"}})
    )

    assert scraper.urls[:3] == [
        "https://notion.so",
        "https://notion.so/pricing",
        "https://notion.so/about",
    ]
    assert len(result["scraped_content"]) == 5
    assert result["scraped_content"][0]["content"] == "x" * 10
    assert result["scraped_content"][0]["truncated"] is True


def test_scrape_worker_records_url_errors_without_stopping() -> None:
    class PartiallyFailingScraper:
        def scrape(self, url: str, user_agent: str) -> dict:
            if "pricing" in url:
                raise RuntimeError("blocked")
            return {"url": url, "title": "OK", "content": "content"}

    worker = ScrapeWorker(
        scraper=PartiallyFailingScraper(),
        min_delay=0,
        max_delay=0,
    )

    result = asyncio.run(
        worker({"competitor": {"name": "Slack", "website": "https://slack.com"}})
    )

    errors = [item for item in result["scraped_content"] if item.get("error")]
    successes = [item for item in result["scraped_content"] if not item.get("error")]
    assert len(errors) == 1
    assert "blocked" in errors[0]["message"]
    assert successes


def test_scrape_worker_uses_review_sites_without_website() -> None:
    worker = ScrapeWorker(min_delay=0, max_delay=0)

    urls = worker._build_target_urls("Microsoft Teams", None)

    assert urls == [
        "https://www.g2.com/search?query=Microsoft+Teams",
        "https://www.capterra.com/search/?query=Microsoft+Teams",
    ]


def test_qdrant_store_ingests_and_searches_documents() -> None:
    store = QdrantStore(collection_name="test_compintel")
    count = store.ingest(
        [
            RagDocument(
                text="Notion has a flexible workspace product for docs and project collaboration.",
                source="seed:notion",
                metadata={"competitor": "Notion"},
            ),
            RagDocument(
                text="Slack focuses on team messaging, channels, and collaboration workflows.",
                source="seed:slack",
                metadata={"competitor": "Slack"},
            ),
        ]
    )

    results = store.similarity_search("team collaboration messaging", top_k=2)

    assert count == 2
    assert len(results) == 2
    assert all("text" in result for result in results)
    assert all("score" in result for result in results)


def test_qdrant_store_chunks_long_documents() -> None:
    store = QdrantStore(
        collection_name="test_compintel_chunks",
        chunk_size=20,
        chunk_overlap=5,
    )
    count = store.ingest(
        [
            RagDocument(
                text=" ".join(["chunkable"] * 20),
                source="seed:long",
            )
        ]
    )

    assert count > 1


def test_seed_report_loader_loads_default_saas_reports() -> None:
    store = QdrantStore(collection_name="test_compintel_seed")
    loader = SeedReportLoader(store=store)

    count = loader.load_seed_reports()
    results = store.similarity_search("project management collaboration", top_k=3)

    assert count == 12
    assert len(results) == 3
