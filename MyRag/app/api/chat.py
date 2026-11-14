"""Chat endpoints: one-shot JSON answer plus the recent message list."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from ..config import get_settings
from ..db import session_scope
from ..deps import MachineNotFound, WorkflowNotFound, machine_guard, workflow_guard
from ..llm.query import QueryError, run_query, stream_query
from ..schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/machines", tags=["chat"])
log_router = APIRouter(tags=["chat"])

_CALLS_SQL = text(
    "SELECT * FROM query_llm_call WHERE query_id = :qid ORDER BY call_index"
)


def _machine_embed(machine: dict) -> str:
    return machine["embed_model_id"]


def _ensure_embed_ready(machine: dict) -> None:
    """A machine whose embed model/dim no longer matches the server cannot answer.

    The client is expected to offer a re-embed and retry; this guard makes the API safe
    even when called directly.
    """
    settings = get_settings()
    model_id = machine["embed_model_id"]
    embed_dim = int(machine["embed_dim"])

    if embed_dim == settings.embed_dim and model_id in settings.embed_prices:
        return

    raise HTTPException(
        status_code=409,
        detail={
            "code": "embed_model_stale",
            "machine_id": int(machine["id"]),
            "machine_embed_model_id": model_id,
            "machine_embed_dim": embed_dim,
            "server_embed_dim": settings.embed_dim,
            "suggested_embed_model": settings.default_embed_model,
        },
    )


def _guard_chat(machine_id: int, payload: ChatRequest):
    try:
        machine = machine_guard(machine_id)
        workflow = workflow_guard(payload.workflow_id)
    except MachineNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except WorkflowNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    _ensure_embed_ready(machine)

    settings = get_settings()
    if payload.llm_model_id not in settings.llm_prices:
        known = ", ".join(sorted(settings.llm_prices))
        raise HTTPException(
            status_code=422,
            detail=f"Unknown llm_model_id '{payload.llm_model_id}'. Configured: {known}",
        )
    return machine, workflow


@router.post("/{machine_id}/chat", response_model=ChatResponse)
async def chat(machine_id: int, payload: ChatRequest) -> ChatResponse:
    machine, workflow = _guard_chat(machine_id, payload)

    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    question = payload.question.strip()

    try:
        outcome = await run_query(
            machine_id=machine_id,
            workflow_id=int(workflow["id"]),
            workflow_version=int(workflow["id"]),
            workflow_class_key=workflow["class_key"],
            embed_model_id=_machine_embed(machine),
            llm_model_id=payload.llm_model_id,
            question=question,
            top_k=top_k,
        )
    except QueryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    with session_scope() as session:
        calls = session.execute(_CALLS_SQL, {"qid": outcome.query_id}).mappings().all()

    return ChatResponse(
        query_id=outcome.query_id,
        answer=outcome.answer,
        reasoning=outcome.reasoning,
        workflow_name=workflow["name"],
        llm_model_id=payload.llm_model_id,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        total_tokens=outcome.total_tokens,
        cost_usd=outcome.cost_usd,
        cost_source=outcome.cost_source,
        retrieval_ms=outcome.retrieval_ms,
        total_ms=outcome.total_ms,
        retrieved_chunk_ids=outcome.retrieved_chunk_ids,
        llm_calls=[
            {
                "model_id": row["model_id"],
                "purpose": row["purpose"],
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "total_tokens": int(row["total_tokens"]),
                "cost_usd": float(row["cost_usd"]),
                "cost_source": row["cost_source"],
            }
            for row in calls
        ],
    )


@router.post("/{machine_id}/chat/stream")
async def chat_stream(machine_id: int, payload: ChatRequest) -> StreamingResponse:
    """Same as /chat, but streams reasoning and answer tokens as they arrive."""
    machine, workflow = _guard_chat(machine_id, payload)

    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    question = payload.question.strip()

    async def events():
        try:
            async for event in stream_query(
                machine_id=machine_id,
                workflow_id=int(workflow["id"]),
                workflow_version=int(workflow["id"]),
                workflow_class_key=workflow["class_key"],
                embed_model_id=_machine_embed(machine),
                llm_model_id=payload.llm_model_id,
                question=question,
                top_k=top_k,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Streaming chat failed for machine %s", machine_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@log_router.get("/api/queries/{query_id}/log")
def query_log(query_id: int) -> dict:
    """Retrieval and prompt trace recorded while the query ran."""
    with session_scope() as session:
        found = session.execute(
            text("SELECT id FROM query WHERE id = :id"), {"id": query_id}
        ).mappings().first()
        if found is None:
            raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
        rows = session.execute(
            text(
                "SELECT kind, payload, created_at FROM query_event "
                "WHERE query_id = :id ORDER BY id"
            ),
            {"id": query_id},
        ).mappings().all()

    events = []
    for row in rows:
        payload = row["payload"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {"raw": payload}
        events.append(
            {
                "kind": row["kind"],
                "payload": payload or {},
                "created_at": row["created_at"],
            }
        )

    return {"query_id": query_id, "events": events}


@router.get("/{machine_id}/messages")
def list_messages(machine_id: int, limit: int = 50) -> list[dict]:
    limit = max(1, min(limit, 200))
    with session_scope() as session:
        machine = session.execute(
            text("SELECT id FROM machines WHERE id = :id"), {"id": machine_id}
        ).mappings().first()
        if machine is None:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")

        rows = session.execute(
            text(
                """
                SELECT q.id, q.question, q.answer, q.reasoning, q.llm_model_id,
                       q.prompt_tokens, q.completion_tokens, q.total_tokens,
                       q.cost_usd, q.cost_source, q.retrieval_ms, q.total_ms,
                       q.retrieved_chunk_ids, q.error, q.created_at,
                       w.name AS workflow_name, q.workflow_version
                  FROM query q
                  JOIN workflows w ON w.id = q.workflow_id
                 WHERE q.machine_id = :machine_id
                 ORDER BY q.id DESC
                 LIMIT :limit
                """
            ),
            {"machine_id": machine_id, "limit": limit},
        ).mappings().all()

    return [
        {
            "query_id": int(row["id"]),
            "question": row["question"],
            "answer": row["answer"],
            "reasoning": row["reasoning"],
            "llm_model_id": row["llm_model_id"],
            "workflow_name": row["workflow_name"],
            "workflow_version": int(row["workflow_version"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "total_tokens": int(row["total_tokens"]),
            "cost_usd": float(row["cost_usd"]),
            "cost_source": row["cost_source"],
            "retrieval_ms": row["retrieval_ms"],
            "total_ms": row["total_ms"],
            "retrieved_chunk_ids": list(row["retrieved_chunk_ids"] or []),
            "error": row["error"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]