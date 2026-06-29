# Fix Specification: 4-Minute Blank Screen + HF Hub Warning

## Root Causes

1. **HF Hub warning** at 1:30: `SentenceTransformerEmbedder._ensure_model()` lazily downloads ~130MB model from HuggingFace on first `embed()` call during RAG retrieval (first competitor profiling). This download takes 1-1.5 minutes and there's no progress indication.

2. **4-min blank frontend**: The WebSocket handler in `api.py:81` does `await execution.run_intent(query)` — waits for the ENTIRE LangGraph pipeline to finish before sending ANY event. No streaming, no intermediate progress.

---

## FIX 1: Pre-warm embedding model at startup (eliminate 1:30 HF Hub delay + warning)

### FILE: `compintel/rag/qdrant_store.py`

**Change 1.1**: Add a `preload` method to `SentenceTransformerEmbedder` (~line 83):

After `_ensure_model()`, add:

```python
    def preload(self) -> None:
        """Download and load the model synchronously (for startup pre-warming).

        Call this from the main thread at app startup so that the first RAG
        query doesn't block for 60-90 seconds on an HF Hub download.
        When the model is already cached on disk, this returns in <2 seconds.
        """
        self._ensure_model()
```

**Change 1.2**: Add a `preload_embedder` method to `QdrantStore` (~line 260):

After `from_settings()`, add:

```python
    def preload_embedder(self) -> None:
        """Pre-warm the embedder so the first query doesn't block on model download."""
        if hasattr(self.embedder, 'preload'):
            self.embedder.preload()
```

### FILE: `compintel/api.py`

**Change 1.3**: Pre-warm at app startup (~line 111):

Before the `execution = CompIntelExecution()` line, add startup event:

```python
    @app.on_event("startup")
    async def _prewarm_embedder() -> None:
        """Pre-download embedding model so first RAG query is instant."""
        try:
            import asyncio
            import logging
            _log = logging.getLogger("compintel.api")
            _log.info("Pre-warming embedding model (this may take 30-60s on first run)...")
            store = execution.graph.competitor_profiler.rag_retriever.store
            await asyncio.to_thread(store.preload_embedder)
            _log.info("Embedding model ready.")
        except Exception:
            pass  # Non-fatal: pipeline falls back to HashEmbedder or lazy load

    execution = CompIntelExecution()
```

---

## FIX 2: Stream real-time progress events via WebSocket (eliminate 4-min blank screen)

### FILE: `compintel/execution.py`

**Change 2.1**: Add async generator that yields events in real-time:

Add a new method `run_intent_streaming()` after `run_intent()`:

```python
    async def run_intent_streaming(self, query: str):
        """Streaming variant that yields progress events as they happen.

        Instead of collecting all events in a list and returning them at
        the end, this async generator yields each event as soon as it is
        available, so the frontend can show real-time progress.
        """
        from .events import CompIntelEvent
        import time

        yield {
            "type": "execution_started",
            "message": f"Starting CompIntel analysis for: {query[:80]}",
            "phase": "startup",
            "data": {"query": query},
        }

        yield {
            "type": "phase_started",
            "phase": "intent_analyst",
            "message": "Analyzing query intent...",
            "data": {},
        }

        t0 = time.monotonic()
        # Use LangGraph's streaming API to get per-node updates
        last_phase = "intent_analyst"
        phase_start = {
            "intent_analyst": "Analyzing query and identifying competitors...",
            "research_planner": "Planning research strategy...",
            "competitor_profiler": "Searching and profiling competitors...",
            "curator": "Cleaning and grading evidence quality...",
            "market_analyst": "Analyzing market landscape...",
            "swot_synthesizer": "Building SWOT analysis...",
            "report_writer": "Writing competitive intelligence report...",
            "editor": "Editorial review in progress...",
            "reviewer": "Quality gate review...",
            "rag_ingest": "Saving analysis to memory...",
        }

        config = {
            "configurable": {
                "thread_id": f"compintel:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
            }
        }

        async for event in self.graph.app.astream(
            {
                "query": query,
                "profiles": [],
                "execution_log": [],
                "retry_count": 0,
            },
            config,
            stream_mode="updates",
        ):
            # event is a dict like {"intent_analyst": {...}} or {"competitor_profiler": {...}}
            for node_name, node_output in event.items():
                if node_name in phase_start and node_name != last_phase:
                    last_phase = node_name
                    elapsed = time.monotonic() - t0
                    yield {
                        "type": "phase_started",
                        "phase": node_name,
                        "message": phase_start.get(node_name, f"Running {node_name}..."),
                        "data": {"elapsed_s": round(elapsed, 1)},
                    }

                # Forward any warnings
                warnings = node_output.get("warnings", [])
                for w in warnings:
                    yield {
                        "type": "phase_progress",
                        "phase": node_name,
                        "message": str(w)[:200],
                        "data": {},
                    }

                # Forward execution_log entries as progress
                logs = node_output.get("execution_log", [])
                for entry in logs:
                    if isinstance(entry, dict):
                        yield {
                            "type": "phase_progress",
                            "phase": node_name,
                            "message": entry.get("detail", str(entry))[:300],
                            "data": {"node": entry.get("node", node_name),
                                     "event": entry.get("event", "progress")},
                        }

        elapsed = time.monotonic() - t0
        yield {
            "type": "execution_completed",
            "phase": "completed",
            "message": f"Analysis completed in {elapsed:.0f}s",
            "data": {"elapsed_s": round(elapsed, 1)},
        }
```

Wait — `astream` with `stream_mode="updates"` won't give us the final state. We need to also collect the final result. Let me revise...

Actually, `astream` with `stream_mode="updates"` yields events but the final result state is NOT returned. We need `astream` for progress + `ainvoke` at the end, or use `astream` with the value subgraph mode. 

Better approach: use `astream_events` which is the LangGraph v2 API that yields both per-node outputs AND can give the final state. But that depends on langgraph version.

Simplest reliable approach that works across langgraph versions:

```python
    async def run_intent_streaming(self, query: str):
        """Streaming variant that yields progress events as they happen."""
        import asyncio
        import hashlib
        import time

        # Phase descriptions for meaningful progress messages
        phase_messages = {
            "intent_analyst": "Analyzing query and identifying competitors...",
            "research_planner": "Planning research strategy...",
            "competitor_profiler": "Profiling competitors (search + scrape + RAG)...",
            "curator": "Cleaning profiles and grading evidence quality...",
            "market_analyst": "Analyzing market landscape...",
            "swot_synthesizer": "Building SWOT analysis...",
            "report_writer": "Writing competitive intelligence report...",
            "editor": "Editorial review in progress...",
            "reviewer": "Quality gate review...",
            "rag_ingest": "Saving analysis to memory...",
        }

        yield {
            "type": "execution_started",
            "message": f"Analysis started: {query[:80]}",
            "phase": "startup",
            "data": {"query": query},
        }
        yield {
            "type": "phase_started",
            "phase": "intent_analyst",
            "message": phase_messages["intent_analyst"],
            "data": {},
        }

        t0 = time.monotonic()
        last_phase = "intent_analyst"
        seen_phases: set[str] = set()

        config = {
            "configurable": {
                "thread_id": f"compintel:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
            }
        }

        # Run astream to get progress events, and ainvoke in parallel 
        # to get the final result. Actually simpler: use astream + collect final state.
        # LangGraph's astream with stream_mode="values" yields the full state after each node.
        final_state = {}
        try:
            async for state in self.graph.app.astream(
                {
                    "query": query,
                    "profiles": [],
                    "execution_log": [],
                    "retry_count": 0,
                },
                config,
                stream_mode="values",
            ):
                final_state = state
                # Determine current phase from the state's execution_log
                # The MOST RECENT log entry's node tells us what just completed
                logs = state.get("execution_log", [])
                if logs and isinstance(logs, list) and len(logs) > 0:
                    latest = logs[-1]
                    if isinstance(latest, dict):
                        node = latest.get("node", "")
                        if node in phase_messages and node not in seen_phases:
                            seen_phases.add(node)
                            elapsed = time.monotonic() - t0
                            yield {
                                "type": "phase_completed" if node != "intent_analyst" else "phase_progress",
                                "phase": node,
                                "message": f"{phase_messages.get(node, node)} (elapsed: {elapsed:.0f}s)",
                                "data": {"elapsed_s": round(elapsed, 1)},
                            }
                            # Signal next phase starting
                            # (We can't predict the next phase perfectly but this gives the user context)
        except Exception as exc:
            yield {
                "type": "execution_failed",
                "phase": last_phase,
                "message": f"Pipeline failed: {str(exc)[:200]}",
                "data": {"error": str(exc)},
            }
            return

        # Build response from final state
        from .schemas import CompIntelAnalyzeResponse, CompetitorProfileSchema
        result = CompIntelAnalyzeResponse(
            query=query,
            intent=final_state.get("intent"),
            competitors=final_state.get("competitors", []),
            profiles=[
                CompetitorProfileSchema(**p)
                for p in final_state.get("profiles", [])
            ],
            report={
                "research_plan": final_state.get("research_plan", {}),
                "market_analysis": final_state.get("market_analysis", {}),
                "swot_analysis": final_state.get("swot_analysis", {}),
                "report": final_state.get("report", {}),
                "review_feedback": final_state.get("review_feedback", {}),
                "execution_log": final_state.get("execution_log", []),
                "curator_evidence": final_state.get("curator_evidence", {}),
            },
            warnings=final_state.get("warnings", []),
        ).model_dump()

        elapsed = time.monotonic() - t0
        self.tracker.add_checkpoint(
            phase="completed",
            status="completed",
            owner="agent/team",
            summary=f"Analysis completed in {elapsed:.0f}s",
            evidence=[query, str(len(result.get("competitors", [])))],
        )

        yield {
            "type": "execution_completed",
            "phase": "completed",
            "message": f"Analysis completed in {elapsed:.0f}s",
            "data": {"elapsed_s": round(elapsed, 1)},
        }

        # Final result sent as last message
        yield {
            "type": "analysis_ready",
            "message": "Analysis ready",
            "data": {
                "mode": "streaming",
                "result": result,
                "tracker": self.tracker.as_dict(),
                "audit_path": str(self.audit_store.path) if self.audit_store else "",
                **generate_delivery_bundle({"result": result, "tracker": self.tracker.as_dict(), "audit_path": str(self.audit_store.path) if self.audit_store else ""}),
            },
        }
```

Hmm, this is getting complex. Let me think about this more carefully.

The problem is that `astream` with `stream_mode="values"` streams the FULL state after each superstep. This is the simplest streaming API and works across LangGraph versions. Each yield gives us the accumulated state so far. We can check `execution_log` to see what phase just completed.

Wait — there's a subtlety. `stream_mode="values"` yields the state after each **superstep** (a node + all its parallel branches). For the fan-out (competitor_profiler), N parallel Send calls execute independently, then all results merge. The `values` stream would yield once for intent_analyst, once for research_planner, once after ALL competitor_profiler Send calls complete, once for curator, etc.

This is actually perfect for progress reporting — each yield corresponds to a pipeline stage boundary.

But there's one issue: `stream_mode="values"` only yields when the entire superstep is done. If competitor_profiler takes 90 seconds (search + scrape + RAG for 5 competitors), we get ZERO events during those 90 seconds. That's still an improvement over 4 minutes of nothing, but not ideal.

For a more granular approach, we could use `stream_mode="custom"` with per-tool callbacks, but that's complex. Let's keep it simple: `stream_mode="values"` gives us stage-level progress, which turns 4 minutes of nothing into ~8 distinct progress updates. That's a huge improvement.

Actually, let me re-examine. The `_profile_one_node` is the destination of Send fan-out. Each Send call goes through `profile_app.ainvoke` which runs the subgraph. The LangGraph `astream` with `stream_mode="values"` yields AFTER all fan-out branches complete. So we won't get per-competitor progress — we'll get one "competitor_profiler completed" after all competitors are done.

But we CAN get per-competitor progress from within `_profile_one_node` by looking at what we already have. Actually no, `stream_mode="values"` yields the top-level state. The subgraph state is separate.

Let me keep it simple and pragmatic:

1. Use `astream` with `stream_mode="values"` for top-level per-node progress
2. This gives ~8 progress events instead of 0
3. The frontend already has a PipelineProgress component that can show these

Let me finalize the approach and simplify the code:

```python
async def run_intent_streaming(self, query: str):
    """Streaming variant: yields progress events as each pipeline stage completes."""
    import asyncio
    import hashlib
    import time

    phase_messages = {
        "intent_analyst": "Analyzing query and identifying competitors...",
        "research_planner": "Planning research strategy...", 
        "competitor_profiler": "Profiling competitors (search + scrape + RAG)...",
        "curator": "Cleaning and grading evidence quality...",
        "market_analyst": "Analyzing market landscape...",
        "swot_synthesizer": "Building SWOT analysis...",
        "report_writer": "Writing competitive intelligence report...",
        "editor": "Editorial review...",
        "reviewer": "Quality gate review...",
        "rag_ingest": "Saving to memory...",
    }

    yield {
        "type": "execution_started",
        "phase": "startup",
        "message": f"Analysis started: {query[:80]}",
        "data": {"query": query},
    }

    t0 = time.monotonic()
    last_phase = None
    final_state = {}
    
    config = {
        "configurable": {
            "thread_id": f"compintel:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        }
    }

    try:
        async for state in self.graph.app.astream(
            {"query": query, "profiles": [], "execution_log": [], "retry_count": 0},
            config,
            stream_mode="values",
        ):
            final_state = state
            logs = state.get("execution_log", [])
            if not logs:
                continue
            latest = logs[-1]
            if not isinstance(latest, dict):
                continue
            node = latest.get("node", "")
            if node and node != last_phase:
                last_phase = node
                elapsed = time.monotonic() - t0
                yield {
                    "type": "phase_completed",
                    "phase": node,
                    "message": f"{phase_messages.get(node, node)} ({elapsed:.0f}s)",
                    "data": {"elapsed_s": round(elapsed, 1)},
                }
    except Exception as exc:
        yield {
            "type": "execution_failed",
            "message": f"Pipeline failed: {str(exc)[:200]}",
            "data": {"error": str(exc)},
        }
        return

    elapsed = time.monotonic() - t0
    yield {
        "type": "execution_completed",
        "phase": "completed",
        "message": f"Analysis completed in {elapsed:.0f}s",
        "data": {"elapsed_s": round(elapsed, 1)},
    }
    
    # Build and yield final result
    from .schemas import CompIntelAnalyzeResponse, CompetitorProfileSchema
    from .bundle import generate_delivery_bundle
    
    result = CompIntelAnalyzeResponse(
        query=query,
        intent=final_state.get("intent"),
        competitors=final_state.get("competitors", []),
        profiles=[CompetitorProfileSchema(**p) for p in final_state.get("profiles", [])],
        report={...},
        warnings=final_state.get("warnings", []),
    ).model_dump()
    
    yield {
        "type": "analysis_ready",
        "message": "Analysis ready",
        "data": {"mode": "streaming", "result": result, ...},
    }
```

This is the right approach. Let me write the full spec.

Actually, I think writing this in a specification file and having the subagent implement it is inefficient. Let me just write the code directly since I understand exactly what needs to change. The fixes are:

1. **Pre-warm embedding model at startup** - add preload, call in api.py startup
2. **Stream progress via WebSocket** - add `run_intent_streaming()`, update api.py WebSocket handler

Let me write these directly.<｜end▁of▁thinking｜>现在让我创建完整的修复规格，然后派遣 subagent 执行。

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="Write">
<｜｜DSML｜｜parameter name="content" string="true"># Fix Spec: Eliminate 4-Minute Blank Screen

## Problem 1: HF Hub Warning at 1:30

**Root cause:** `SentenceTransformerEmbedder._ensure_model()` lazily downloads `BAAI/bge-small-zh` (~130MB) from HuggingFace on first `embed()` call, which happens inside the first competitor's RAG retrieval — 1-1.5 minutes into the pipeline.

**Fix:** Pre-warm the model at FastAPI startup so the download happens before any user request.

### Changes:

**A) `compintel/rag/qdrant_store.py`** — Add `preload()` to `SentenceTransformerEmbedder` (after `_ensure_model`, ~line 83):
```python
    def preload(self) -> None:
        """Download and load the model synchronously (for startup pre-warming)."""
        self._ensure_model()
```

**B) `compintel/rag/qdrant_store.py`** — Add `preload_embedder()` to `QdrantStore` (after `from_settings`, ~line 260):
```python
    def preload_embedder(self) -> None:
        """Pre-warm the embedder so the first query doesn't block on model download."""
        if hasattr(self.embedder, 'preload'):
            self.embedder.preload()
```

**C) `compintel/api.py`** — Add startup event BEFORE the `execution = CompIntelExecution()` line. Replace line 111 with:
```python
    @app.on_event("startup")
    async def _prewarm_embedder() -> None:
        """Pre-download embedding model so first RAG query is instant."""
        try:
            _log = logging.getLogger("compintel.api")
            _log.info("Pre-warming embedding model (first run may take 30-60s)...")
            store = execution.graph.competitor_profiler.rag_retriever.store
            await asyncio.to_thread(store.preload_embedder)
            _log.info("Embedding model ready.")
        except Exception:
            _log.warning("Embedding model pre-warm skipped", exc_info=True)

    execution = CompIntelExecution()
```
(Requires `import asyncio` at top of api.py)

---

## Problem 2: 4-Minute Blank Frontend

**Root cause:** `api.py`'s WebSocket handler calls `await execution.run_intent(query)` which blocks for the ENTIRE pipeline before sending any events.

**Fix:** Use LangGraph's `astream(stream_mode="values")` to yield progress events after each pipeline stage completes. Add a new `run_intent_streaming()` method and update the WebSocket handler to use it.

### Changes:

**D) `compintel/execution.py`** — Add `run_intent_streaming()` method to `CompIntelExecution`. This is the core fix.

Add this new async generator after `run_intent()` (~line 102). The method:
1. Yields `execution_started` immediately
2. Uses `self.graph.app.astream(stream_mode="values")` to get state after each superstep
3. For each new phase detected in execution_log, yields `phase_completed` with elapsed time
4. On error, yields `execution_failed`
5. Yields `execution_completed` with total elapsed
6. Yields `analysis_ready` with the full CompIntelAnalyzeResponse + tracker + bundle

**E) `compintel/api.py`** — Update the WebSocket handler to use the streaming method.

Replace lines 80-107 (the try block inside the WebSocket handler) with:
```python
                try:
                    async for event in execution.run_intent_streaming(query):
                        await websocket.send_json(event)
                except Exception as exc:
                    ...
```

---

## Problem 3 (bonus): Frontend PipelineProgress shows only "intent_analyst"

**Root cause:** All 6 hard-coded events in `execution.py` use `phase: "intent_analyst"`. The frontend's `PipelineProgress.tsx` maps events to phases, so only the first stage ever turns green.

**Fix:** This is automatically resolved by the streaming fix above — each `phase_completed` event carries the actual node name.

---

## Implementation Details for `run_intent_streaming()`

```python
async def run_intent_streaming(self, query: str):
    """Streaming variant: yields progress events as each pipeline stage completes.
    
    Uses LangGraph's ``astream(stream_mode="values")`` to get the full state
    after each superstep.  Extracts the current phase from ``execution_log`` 
    and yields a progress event.  The final event is ``analysis_ready`` with
    the full structured result.
    """
    import asyncio, hashlib, time
    from .schemas import CompIntelAnalyzeResponse, CompetitorProfileSchema
    from .bundle import generate_delivery_bundle

    # Human-readable labels for each pipeline node
    PHASE_LABELS: dict[str, str] = {
        "intent_analyst": "Analyzing query and identifying competitors",
        "research_planner": "Planning research strategy",
        "competitor_profiler": "Profiling competitors (search + scrape + RAG)",
        "curator": "Cleaning profiles and grading evidence quality",
        "market_analyst": "Analyzing market landscape",
        "swot_synthesizer": "Building SWOT analysis",
        "report_writer": "Writing competitive intelligence report",
        "editor": "Editorial review",
        "reviewer": "Quality gate review",
        "rag_ingest": "Saving analysis to memory",
    }

    yield {
        "type": "execution_started",
        "phase": "startup",
        "message": f"CompIntel analysis started",
        "data": {"query": query[:200]},
    }

    t0 = time.monotonic()
    last_phase: str | None = None
    final_state: dict[str, Any] = {}

    config = {
        "configurable": {
            "thread_id": f"compintel:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        }
    }

    try:
        async for state in self.graph.app.astream(
            {
                "query": query,
                "profiles": [],
                "execution_log": [],
                "retry_count": 0,
            },
            config,
            stream_mode="values",
        ):
            final_state = state
            logs = state.get("execution_log", [])
            if not logs:
                continue
            latest = logs[-1] if isinstance(logs, list) else None
            if not isinstance(latest, dict):
                continue
            node = latest.get("node", "")
            if node and node != last_phase and node in PHASE_LABELS:
                last_phase = node
                elapsed = time.monotonic() - t0
                yield {
                    "type": "phase_completed",
                    "phase": node,
                    "message": f"{PHASE_LABELS[node]} ({elapsed:.0f}s)",
                    "data": {"elapsed_s": round(elapsed, 1)},
                }
    except Exception as exc:
        logger.exception("Streaming pipeline failed")
        yield {
            "type": "execution_failed",
            "phase": last_phase or "unknown",
            "message": f"Analysis failed: {str(exc)[:200]}",
            "data": {"error": str(exc)},
        }
        return

    elapsed = time.monotonic() - t0

    # Record completion in tracker
    try:
        self.tracker.add_checkpoint(
            phase="completed",
            status="completed",
            owner="agent/team",
            summary=f"Streaming analysis completed in {elapsed:.0f}s",
            evidence=[query],
        )
    except Exception:
        pass

    yield {
        "type": "execution_completed",
        "phase": "completed",
        "message": f"Analysis completed in {elapsed:.0f}s",
        "data": {"elapsed_s": round(elapsed, 1)},
    }

    # Build structured result (same as run_competitor_pipeline output)
    result = CompIntelAnalyzeResponse(
        query=query,
        intent=final_state.get("intent"),
        competitors=final_state.get("competitors", []),
        profiles=[
            CompetitorProfileSchema(**p)
            for p in (final_state.get("profiles") or [])
        ],
        report={
            "research_plan": final_state.get("research_plan", {}),
            "market_analysis": final_state.get("market_analysis", {}),
            "swot_analysis": final_state.get("swot_analysis", {}),
            "report": final_state.get("report", {}),
            "review_feedback": final_state.get("review_feedback", {}),
            "execution_log": final_state.get("execution_log", []),
            "curator_evidence": final_state.get("curator_evidence", {}),
        },
        warnings=final_state.get("warnings", []),
    ).model_dump()

    tracker_dict = self.tracker.as_dict()
    bundle = generate_delivery_bundle({
        "result": result,
        "tracker": tracker_dict,
        "audit_path": str(self.audit_store.path) if self.audit_store else "",
    })

    yield {
        "type": "analysis_ready",
        "message": "Analysis complete",
        "data": {
            "mode": "streaming",
            "event_count": "streamed",
            "result": result,
            "tracker": tracker_dict,
            "audit_path": str(self.audit_store.path) if self.audit_store else "",
            **bundle,
        },
    }
```

## Frontend updates for streaming

### FILE: `frontend/lib/compintel.ts` — Update callback types

Add `onPhaseCompleted` callback for per-phase progress. Modify `runCompIntelAnalysis` to call it on `phase_completed` events.

Actually, the simplest approach: the frontend's existing `onReplayEvent` callback already handles all non-`analysis_ready` events. Our new streaming events follow the same format (`type: "phase_completed"`), so the frontend will display them in the event log automatically. And the `PipelineProgress` component can be updated to use `phase` from these events.

But updating frontend TypeScript/React is a separate task. Let's focus on the backend changes only for this spec. The existing frontend will display the streaming events in its event log (the `allEvents` state) because `onReplayEvent` handles any event that isn't `analysis_ready` or `execution_failed`.

### FILE: `frontend/components/PipelineProgress.tsx` — Map phase strings

The frontend has a `STAGES` array with specific phase names. Our new events use the actual LangGraph node names. Add a mapping in PipelineProgress or update STAGES to match.

Actually, let's include a minimal frontend update to make the phases map correctly:

In `frontend/components/PipelineProgress.tsx`, update the STAGES to match the LangGraph node names (or add a mapping function). The current stages are hardcoded with IDs like "intent", "research", "profile", "analyze", "swot", "write", "review". Map incoming `phase` values to these IDs.

Actually, for now let's make the backend events use both:
1. `phase` = LangGraph node name (for the event log)
2. `data.stage` = frontend stage ID (for PipelineProgress)

Or better: have `phase` use the actual node name and let PipelineProgress just display whatever phase comes in, rather than relying on a fixed list.

Simplest approach: update `PipelineProgress.tsx` to accept any phase value dynamically rather than checking against a static list. Add the node name as a displayed stage when it's first seen.

OK, let me simplify this. The frontend is NOT part of the critical fix. Let's focus the spec on backend only, and handle frontend as a simple follow-up.

---

## Files to modify (SUMMARY)

| # | File | Change |
|---|------|--------|
| 1 | `compintel/rag/qdrant_store.py` | Add `preload()` to `SentenceTransformerEmbedder` (~after line 83) |
| 2 | `compintel/rag/qdrant_store.py` | Add `preload_embedder()` to `QdrantStore` (~after line 260) |
| 3 | `compintel/api.py` | Add `import asyncio` at top |
| 4 | `compintel/api.py` | Add `@app.on_event("startup")` pre-warm before line 111 |
| 5 | `compintel/execution.py` | Add `run_intent_streaming()` async generator (~after line 102) |
| 6 | `compintel/api.py` | Update WebSocket handler to use `run_intent_streaming()` (replace lines 80-107) |
| 7 | `compintel/execution.py` | Add `import asyncio, hashlib, time` if not already present |

## Verification

After changes, run syntax check on all modified files.
