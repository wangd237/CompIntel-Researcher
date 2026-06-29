"""FastAPI surface for CompIntel Research."""

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .events import CompIntelEvent
from .bundle import generate_delivery_bundle
from .execution import CompIntelExecution
from .schemas import CompIntelAnalyzeRequest, CompIntelAnalyzeResponse

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:

    app = FastAPI(title="CompIntel Research", version="0.1.0")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/report/{bundle_id}/report.md")
    async def download_report(bundle_id: str):
        """Serve the Markdown report file from a completed analysis bundle."""
        from fastapi.responses import FileResponse, PlainTextResponse
        from pathlib import Path
        # Use project-root-relative path, not CWD-relative.
        # uvicorn may be started from a different directory.
        _root = Path(__file__).resolve().parent.parent
        bundle_dir = _root / "outputs" / bundle_id
        report_path = bundle_dir / "report.md"
        if not report_path.exists():
            return PlainTextResponse("Report not found", status_code=404)
        return FileResponse(
            str(report_path),
            media_type="text/markdown; charset=utf-8",
            filename=f"{bundle_id}_report.md",
        )

    @app.get("/api/report/{bundle_id}/{filename:path}")
    async def download_bundle_file(bundle_id: str, filename: str):
        """Serve any file from a completed analysis bundle (snapshot.json, progress.md, etc.)."""
        from fastapi.responses import FileResponse, PlainTextResponse
        from pathlib import Path
        _root = Path(__file__).resolve().parent.parent
        bundle_dir = _root / "outputs" / bundle_id
        file_path = bundle_dir / filename
        # Security: only serve files within the bundle directory
        try:
            file_path = file_path.resolve()
            bundle_dir = bundle_dir.resolve()
            if not str(file_path).startswith(str(bundle_dir)):
                return PlainTextResponse("Forbidden", status_code=403)
        except Exception:
            return PlainTextResponse("Invalid path", status_code=400)
        if not file_path.exists() or not file_path.is_file():
            return PlainTextResponse("File not found", status_code=404)
        # Determine media type
        suffix = file_path.suffix.lower()
        media_type = {
            ".md": "text/markdown; charset=utf-8",
            ".json": "application/json",
            ".txt": "text/plain; charset=utf-8",
        }.get(suffix, "application/octet-stream")
        return FileResponse(
            str(file_path),
            media_type=media_type,
            filename=filename,
        )

    @app.post("/api/compintel/analyze")
    async def analyze(request: CompIntelAnalyzeRequest) -> CompIntelAnalyzeResponse:
        outcome = await execution.run_intent(request.query)
        result = outcome["result"]
        intent = result.get("intent") or {}
        bundle_paths = generate_delivery_bundle(outcome)
        return CompIntelAnalyzeResponse(
            query=request.query,
            intent=intent,
            competitors=result.get("competitors", []),
            profiles=result.get("profiles", []),
            report={
                "tracker": outcome["tracker"],
                "audit_path": outcome["audit_path"],
                "result": result,
                **bundle_paths,
            },
            warnings=result.get("warnings", []) or result.get("notes", []),
        )

    @app.websocket("/ws/compintel")
    async def compintel_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "type": "execution_failed",
                            "message": "invalid json payload",
                            "data": {"raw": message[:200]},
                        }
                    )
                    continue

                query = str(payload.get("query", "")).strip()
                if not query:
                    await websocket.send_json(
                        {
                            "type": "execution_failed",
                            "message": "query is required",
                            "data": {"payload": payload},
                        }
                    )
                    continue

                try:
                    async for event in execution.run_intent_streaming(query):
                        await websocket.send_json(event)
                except Exception as exc:
                    logger.exception("Analysis failed")
                    await websocket.send_json(
                        {
                            "type": "execution_failed",
                            "message": "analysis failed",
                            "data": {"error": str(exc)},
                        }
                    )
        except WebSocketDisconnect:
            return

    execution = CompIntelExecution()

    @app.on_event("startup")
    async def _prewarm_embedder() -> None:
        """Pre-download embedding model in the background so first RAG query is instant.

        Runs as a background task so the server is immediately available for
        health checks and WebSocket connections.  The model download may take
        30-60 seconds on first run (downloading ~130 MB from HuggingFace); if
        the download fails, the pipeline will lazily load the model on first
        RAG query (or fall back to HashEmbedder when EMBEDDING_MODEL is empty).
        """
        _log = logging.getLogger("compintel.api")
        _log.info("Pre-warming embedding model in background (first run downloads ~130 MB)...")

        async def _warm() -> None:
            try:
                store = execution.graph.competitor_profiler.rag_retriever.store
                await asyncio.to_thread(store.preload_embedder)
                _log.info("Embedding model ready.")
            except Exception:
                _log.warning(
                    "Embedding model pre-warm failed — will lazily load on first query",
                    exc_info=True,
                )

        asyncio.create_task(_warm())

    return app


async def _stream_events(websocket: WebSocket, events: list[CompIntelEvent]) -> None:
    for event in events:
        await websocket.send_json(event)
