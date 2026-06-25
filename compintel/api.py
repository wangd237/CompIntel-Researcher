"""FastAPI surface for CompIntel Research."""

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
                    outcome = await execution.run_intent(query)
                except Exception as exc:
                    logger.exception("Analysis failed")
                    await websocket.send_json(
                        {
                            "type": "execution_failed",
                            "message": "analysis failed",
                            "data": {"error": str(exc)},
                        }
                    )
                    continue

                await _stream_events(websocket, outcome["events"])
                await websocket.send_json(
                    {
                        "type": "analysis_ready",
                        "message": "analysis completed",
                        "data": {
                            "mode": "replay",
                            "event_count": len(outcome["events"]),
                            "result": outcome["result"],
                            "tracker": outcome["tracker"],
                            "audit_path": outcome["audit_path"],
                            **generate_delivery_bundle(outcome),
                        },
                    }
                )
        except WebSocketDisconnect:
            return

    execution = CompIntelExecution()
    return app


async def _stream_events(websocket: WebSocket, events: list[CompIntelEvent]) -> None:
    for event in events:
        await websocket.send_json(event)
