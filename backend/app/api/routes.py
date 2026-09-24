"""REST endpoints. Handlers stay thin; logic lives in services/agents."""

import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.agents.registry import list_agents
from app.api.schemas import (
    AgentSummary,
    ChatRequest,
    ChatResponse,
    SessionResponse,
    SessionTurn,
    UploadResponse,
)
from app.config import get_settings
from app.llm.providers import any_model_present, check_llm_ready
from app.observability.context import trace_id_var

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/agents", response_model=list[AgentSummary], tags=["agents"])
def get_agents() -> list[AgentSummary]:
    """List registered agents, what they handle, and the skills they can call."""
    return [AgentSummary(name=a.name, description=a.description,
                         skills=[skill.name for skill in a.skills])
            for a in list_agents()]


@router.post("/upload", response_model=UploadResponse, tags=["documents"])
async def upload(file: UploadFile) -> UploadResponse:
    """Ingest a PDF or CSV/Excel document."""
    from app.ingestion.pipeline import PDF_SUFFIXES, TABULAR_SUFFIXES, ingest_file

    settings = get_settings()
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in PDF_SUFFIXES | TABULAR_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix or '(none)'}; "
                   f"expected one of {sorted(PDF_SUFFIXES | TABULAR_SUFFIXES)}",
        )

    payload = await file.read()
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB")

    destination = Path(settings.upload_dir) / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)

    report = ingest_file(destination, settings)
    if not report.files:
        raise HTTPException(status_code=422, detail=f"Nothing could be ingested from {name}")

    return UploadResponse(
        filename=name,
        doc_type="status_report" if suffix in PDF_SUFFIXES else ", ".join(report.rows_loaded) or "table",
        chunks_indexed=report.chunks_indexed,
    )


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest) -> ChatResponse:
    """Route a question to the right agent and return an answer with citations."""
    # Imported here, like the ingestion pipeline above: keeps the retrieval and
    # model stack off the import path of endpoints that do not need it.
    from app.graph.build import run_chat_graph
    from app.memory.session_store import get_session_store, valid_session_id

    if req.session_id is not None and not valid_session_id(req.session_id):
        raise HTTPException(status_code=422, detail="Malformed session_id")

    store = get_session_store()
    # An unknown but well-formed ID starts a new conversation rather than 404ing:
    # a client that kept an ID across a backend restart should keep working.
    session_id = req.session_id or store.new_session()
    trace_id = trace_id_var.get()
    history = store.history(session_id)
    started = time.perf_counter()

    try:
        # One graph per request: rewrite -> route -> the chosen agent's skill
        # loop. Everything that used to sit between here and an agent -- the
        # follow-up rewrite, the routing decision, the fallbacks -- is a node.
        state = await run_chat_graph(req.question, session_id, history)
        result = state["result"]
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        # The detail is deliberately generic: a provider or DuckDB exception can
        # carry an endpoint URL or a filesystem path. It goes to the log instead.
        logger.exception("chat failed", extra={"fields": {
            "session_id": session_id, "error": f"{type(exc).__name__}: {exc}"}})
        raise HTTPException(status_code=503, detail="The assistant is unavailable") from exc

    store.append(session_id, req.question, result.answer, result.agent)
    logger.info("chat", extra={"fields": {
        "session_id": session_id,
        "agent": result.agent,
        "routing": result.metadata.get("routing"),
        "rewritten": state.get("rewritten", False),
        "retrieved": result.metadata.get("retrieved"),
        "sql": result.metadata.get("sql"),
        "tool_calls": result.metadata.get("tool_calls"),
        "citations": len(result.citations),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }})
    return ChatResponse(answer=result.answer, agent=result.agent,
                        citations=result.citations, session_id=session_id,
                        trace_id=trace_id)


@router.get("/sessions/{session_id}", response_model=SessionResponse, tags=["chat"])
async def get_session(session_id: str) -> SessionResponse:
    """Return the message history for a session."""
    from app.memory.session_store import get_session_store, valid_session_id

    if not valid_session_id(session_id):
        raise HTTPException(status_code=422, detail="Malformed session_id")

    store = get_session_store()
    transcript = store.transcript(session_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")

    return SessionResponse(
        session_id=session_id,
        turns=[SessionTurn(question=t.question, answer=t.answer, agent=t.agent, at=t.at)
               for t in transcript],
        truncated=len(transcript) >= store.max_turns,
    )


@router.get("/health/dependencies", tags=["system"])
def health_dependencies() -> dict:
    """Readiness of the backing services, and what the deployment mode resolved to.

    Separate from `/health`, which is the container healthcheck and must stay
    fast and dependency-free. This is the endpoint to hit after flipping
    DEPLOYMENT_MODE or repointing OLLAMA_BASE_URL at a different machine.
    """
    settings = get_settings()
    checks = {
        "llm": check_llm_ready(settings),
        "vector_store": _check_vector_store(settings),
        "tabular": _check_writable(Path(settings.tabular_db_path).parent),
        "checkpointer": _check_checkpointer(settings),
    }
    # `reachable` alone only says the provider answered, so a chain of models
    # that no longer exist reported `ok` -- which is how the retired
    # `gemini-2.5-flash` default went unnoticed. Both are required now.
    ok = (
        checks["llm"].get("reachable")
        and any_model_present(checks["llm"])
        and all(
            c.get("ok") for c in (checks["vector_store"], checks["tabular"],
                                  checks["checkpointer"])
        )
    )
    return {
        "status": "ok" if ok else "degraded",
        "resolved": settings.resolved_summary(),
        "checks": checks,
    }


def _check_vector_store(settings) -> dict:
    """Check the Qdrant target without taking the embedded-mode directory lock."""
    if settings.qdrant_url:
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
            client.get_collections()
            return {"ok": True, "mode": "server", "target": settings.qdrant_url}
        except Exception as exc:
            return {"ok": False, "mode": "server", "error": f"{type(exc).__name__}: {exc}"}
    result = _check_writable(Path(settings.qdrant_path))
    result |= {"mode": "embedded", "collection": settings.collection_name}
    return result


def _check_checkpointer(settings) -> dict:
    """Ping the checkpoint store.

    Reported even though an unreachable Redis only degrades the graph to the
    in-process saver (D-017): that degradation is exactly what `status:
    "degraded"` is for, and it is otherwise invisible until someone reads a log.
    """
    if not settings.redis_url:
        return {"ok": True, "mode": "in-process"}
    try:
        from redis import Redis

        # Synchronous client deliberately: this endpoint is a `def`, and the
        # graph's own saver is async and bound to the request loop.
        with Redis.from_url(settings.redis_url, socket_connect_timeout=2) as client:
            client.ping()
        return {"ok": True, "mode": "redis", "target": settings.checkpoint_target,
                "ttl_minutes": settings.checkpoint_ttl_minutes}
    except Exception as exc:
        return {"ok": False, "mode": "redis", "target": settings.checkpoint_target,
                "error": f"{type(exc).__name__}: {exc}"}


def _check_writable(path: Path) -> dict:
    """Report whether `path` exists (or can be created) and is writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return {"ok": os.access(path, os.W_OK), "path": str(path)}
    except Exception as exc:
        return {"ok": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
