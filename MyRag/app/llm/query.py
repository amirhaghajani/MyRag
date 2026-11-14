"""Query execution: run a workflow, record usage, return the answer."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy import text

from ..db import session_scope
from ..llm import usage as usage_mod
from ..llm.factory import build_llm
from ..llm.trace import record_prompt, record_retrieval
from ..retrieval import registry
from ..workflows.rag_workflow import RagWorkflow, build_messages

logger = logging.getLogger(__name__)


@dataclass
class QueryOutcome:
    query_id: int
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cost_source: str
    retrieval_ms: int
    total_ms: int
    retrieved_chunk_ids: list[int]
    reasoning: str | None = None


def _insert_query(
    *,
    machine_id: int,
    workflow_id: int,
    workflow_version: int,
    llm_model_id: str,
    question: str,
) -> int:
    with session_scope() as session:
        result = session.execute(
            text(
                """
                INSERT INTO query
                    (machine_id, workflow_id, workflow_version, llm_model_id, question)
                VALUES
                    (:machine_id, :workflow_id, :workflow_version, :llm_model_id, :question)
                RETURNING id
                """
            ),
            {
                "machine_id": machine_id,
                "workflow_id": workflow_id,
                "workflow_version": workflow_version,
                "llm_model_id": llm_model_id,
                "question": question,
            },
        )
        return int(result.scalar_one())


def _save_call(query_id: int, machine_id: int, call_index: int, purpose: str, call) -> None:
    with session_scope() as session:
        session.execute(
            text(
                """
                INSERT INTO query_llm_call
                    (query_id, machine_id, model_id, call_index, purpose,
                     prompt_tokens, completion_tokens, total_tokens,
                     cost_usd, cost_source, usage_raw)
                VALUES
                    (:query_id, :machine_id, :model_id, :call_index, :purpose,
                     :prompt_tokens, :completion_tokens, :total_tokens,
                     :cost_usd, :cost_source, CAST(:usage_raw AS jsonb))
                """
            ),
            {
                "query_id": query_id,
                "machine_id": machine_id,
                "model_id": call.model_id,
                "call_index": call_index,
                "purpose": purpose,
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
                "total_tokens": call.total_tokens,
                "cost_usd": call.cost_usd,
                "cost_source": call.cost_source,
                "usage_raw": json.dumps(call.usage_raw or {}),
            },
        )


def _chunk_ids_param(session, chunk_ids: list[int]):
    """Postgres takes a list for bigint[]; SQLite (tests) needs a JSON string."""
    if session.get_bind().dialect.name == "postgresql":
        return list(chunk_ids)
    return json.dumps(list(chunk_ids))


def _finish_query(
    *,
    query_id: int,
    answer: str,
    calls: list,
    retrieval_ms: int,
    total_ms: int,
    retrieved_chunk_ids: list[int],
    reasoning: str | None = None,
) -> QueryOutcome:
    prompt_tokens = sum(c.prompt_tokens for c in calls)
    completion_tokens = sum(c.completion_tokens for c in calls)
    total_tokens = sum(c.total_tokens for c in calls) or (prompt_tokens + completion_tokens)
    cost_usd = sum(c.cost_usd for c in calls)
    sources = {c.cost_source for c in calls if c.cost_usd > 0}
    if not sources:
        cost_source = "none"
    elif sources == {"provider"}:
        cost_source = "provider"
    elif "estimated" in sources:
        cost_source = "estimated"
    else:
        cost_source = "env"

    with session_scope() as session:
        bind_chunk_ids = _chunk_ids_param(session, retrieved_chunk_ids)
        session.execute(
            text(
                """
                UPDATE query
                   SET answer = :answer,
                       reasoning = :reasoning,
                       retrieval_ms = :retrieval_ms,
                       total_ms = :total_ms,
                       prompt_tokens = :prompt_tokens,
                       completion_tokens = :completion_tokens,
                       total_tokens = :total_tokens,
                       cost_usd = :cost_usd,
                       cost_source = :cost_source,
                       retrieved_chunk_ids = :chunk_ids
                 WHERE id = :query_id
                """
            ),
            {
                "answer": answer,
                "reasoning": reasoning,
                "retrieval_ms": retrieval_ms,
                "total_ms": total_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "cost_source": cost_source,
                "chunk_ids": bind_chunk_ids,
                "query_id": query_id,
            },
        )

    return QueryOutcome(
        query_id=query_id,
        answer=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        cost_source=cost_source,
        retrieval_ms=retrieval_ms,
        total_ms=total_ms,
        retrieved_chunk_ids=retrieved_chunk_ids,
        reasoning=reasoning,
    )


def _fail_query(query_id: int, message: str) -> None:
    with session_scope() as session:
        session.execute(
            text("UPDATE query SET error = :error, answer = :answer WHERE id = :id"),
            {"error": message[:4000], "answer": None, "id": query_id},
        )


async def run_query(
    *,
    machine_id: int,
    workflow_id: int,
    workflow_version: int,
    workflow_class_key: str,
    embed_model_id: str,
    llm_model_id: str,
    question: str,
    top_k: int,
) -> QueryOutcome:
    """Run one question through the selected workflow and persist its cost."""
    started = time.perf_counter()
    query_id = _insert_query(
        machine_id=machine_id,
        workflow_id=workflow_id,
        workflow_version=workflow_version,
        llm_model_id=llm_model_id,
        question=question,
    )

    handler = usage_mod.get_usage_handler()
    token = usage_mod.set_current_query_id(query_id)
    try:
        workflow = RagWorkflow(timeout=120)
        result = await workflow.run(
            question=question,
            machine_id=machine_id,
            workflow_class_key=workflow_class_key,
            embed_model_id=embed_model_id,
            llm_model_id=llm_model_id,
            top_k=top_k,
        )

        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        reasoning = result.get("reasoning") if isinstance(result, dict) else None
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        retrieval_ms = int(result.get("retrieval_ms", 0)) if isinstance(result, dict) else 0
        total_ms = int((time.perf_counter() - started) * 1000)

        calls = handler.drain()
        for index, call in enumerate(calls):
            _save_call(query_id, machine_id, index, "synthesize", call)

        chunk_ids = [int(getattr(c, "chunk_id", 0)) for c in chunks if getattr(c, "chunk_id", None)]

        return _finish_query(
            query_id=query_id,
            answer=answer,
            calls=calls,
            retrieval_ms=retrieval_ms,
            total_ms=total_ms,
            retrieved_chunk_ids=chunk_ids,
            reasoning=reasoning,
        )
    except Exception as exc:
        handler.drain()
        total_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("Query %s failed: %s", query_id, exc, exc_info=True)
        _fail_query(query_id, f"{type(exc).__name__}: {exc}")
        raise QueryError(str(exc), query_id=query_id, elapsed_ms=total_ms) from exc
    finally:
        usage_mod.reset_current_query_id(token)


async def stream_query(
    *,
    machine_id: int,
    workflow_id: int,
    workflow_version: int,
    workflow_class_key: str,
    embed_model_id: str,
    llm_model_id: str,
    question: str,
    top_k: int,
) -> AsyncIterator[dict]:
    """Run one question and yield SSE-style events while the answer streams.

    The query row is created up front so the client can show progress, then
    finalised with the same usage/cost accounting as the non-streaming path.
    """
    started = time.perf_counter()
    query_id = _insert_query(
        machine_id=machine_id,
        workflow_id=workflow_id,
        workflow_version=workflow_version,
        llm_model_id=llm_model_id,
        question=question,
    )

    handler = usage_mod.get_usage_handler()
    token = usage_mod.set_current_query_id(query_id)
    try:
        yield {"type": "meta", "query_id": query_id}
        yield {"type": "status", "stage": "retrieving"}

        retrieval_started = time.perf_counter()
        chunks = registry.retrieve(
            class_key=workflow_class_key,
            machine_id=machine_id,
            question=question,
            top_k=top_k,
            embed_model_id=embed_model_id,
        )
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
        record_retrieval(chunks)

        llm = build_llm(llm_model_id)
        messages = build_messages(question, chunks)
        record_prompt(messages)
        stream = await llm.astream_chat(
            messages,
            stream_options={"include_usage": True},
        )

        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        yield {"type": "status", "stage": "answering"}
        async for response in stream:
            thinking = response.additional_kwargs.get("thinking_delta")
            if thinking:
                reasoning_parts.append(thinking)
                yield {"type": "reasoning", "text": thinking}
            delta = response.delta or ""
            if delta:
                answer_parts.append(delta)
                yield {"type": "token", "text": delta}

        answer = "".join(answer_parts)
        reasoning = "".join(reasoning_parts) or None

        calls = handler.drain()
        for index, call in enumerate(calls):
            _save_call(query_id, machine_id, index, "synthesize", call)

        chunk_ids = [int(getattr(c, "chunk_id", 0)) for c in chunks if getattr(c, "chunk_id", None)]

        outcome = _finish_query(
            query_id=query_id,
            answer=answer,
            calls=calls,
            retrieval_ms=retrieval_ms,
            total_ms=int((time.perf_counter() - started) * 1000),
            retrieved_chunk_ids=chunk_ids,
            reasoning=reasoning,
        )
        yield {
            "type": "done",
            "query_id": outcome.query_id,
            "answer": outcome.answer,
            "reasoning": outcome.reasoning,
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "total_tokens": outcome.total_tokens,
            "cost_usd": outcome.cost_usd,
            "cost_source": outcome.cost_source,
            "retrieval_ms": outcome.retrieval_ms,
            "total_ms": outcome.total_ms,
            "retrieved_chunk_ids": outcome.retrieved_chunk_ids,
        }
    except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
        handler.drain()
        total_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("Streaming query %s failed: %s", query_id, exc, exc_info=True)
        _fail_query(query_id, f"{type(exc).__name__}: {exc}")
        yield {
            "type": "error",
            "query_id": query_id,
            "message": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": total_ms,
        }
    finally:
        usage_mod.reset_current_query_id(token)


class QueryError(RuntimeError):
    def __init__(self, message: str, *, query_id: int, elapsed_ms: int) -> None:
        super().__init__(message)
        self.query_id = query_id
        self.elapsed_ms = elapsed_ms