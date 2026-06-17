from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from compintel.execution import CompIntelExecution
from compintel.bundle import BundleWriter, generate_delivery_bundle
from compintel.export import MarkdownFormatter
from compintel.progress import ProgressSummaryFormatter
from compintel.settings import CompIntelSettings
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
