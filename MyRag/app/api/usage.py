"""Usage reporting: per-machine totals, query detail, workflow comparison."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from ..db import session_scope
from ..schemas import WorkflowComparisonRow

router = APIRouter(prefix="/api", tags=["usage"])

_BY_MODEL_SQL = text(
    """
    SELECT model_id,
           count(*)            AS calls,
           sum(prompt_tokens)  AS prompt_tokens,
           sum(completion_tokens) AS completion_tokens,
           sum(total_tokens)   AS total_tokens,
           sum(cost_usd)       AS cost_usd,
           min(created_at)     AS first_call,
           max(created_at)     AS last_call
      FROM query_llm_call
     WHERE machine_id = :machine_id
     GROUP BY model_id
     ORDER BY cost_usd DESC
    """
)

_BY_WORKFLOW_SQL = text(
    """
    SELECT q.workflow_id,
           w.name AS workflow_name,
           count(*)                  AS runs,
           coalesce(avg(q.cost_usd), 0)        AS avg_cost_usd,
           coalesce(avg(q.total_tokens), 0)    AS avg_total_tokens,
           coalesce(avg(q.total_ms), 0)        AS avg_latency_ms,
           coalesce(sum(q.cost_usd), 0)        AS total_cost_usd,
           max(q.created_at)                   AS last_run_at
      FROM query q
      JOIN workflows w ON w.id = q.workflow_id
     WHERE q.machine_id = :machine_id
     GROUP BY q.workflow_id, w.name
     ORDER BY total_cost_usd DESC
    """
)

_TOTALS_SQL = text(
    """
    SELECT coalesce(sum(prompt_tokens), 0)      AS prompt_tokens,
           coalesce(sum(completion_tokens), 0)  AS completion_tokens,
           coalesce(sum(total_tokens), 0)       AS total_tokens,
           coalesce(sum(cost_usd), 0)           AS cost_usd,
           count(*)                             AS queries
      FROM query
     WHERE machine_id = :machine_id
       AND error IS NULL
    """
)

_INGEST_TOTALS_SQL = text(
    """
    SELECT coalesce(sum(char_count), 0)     AS char_count,
           coalesce(sum(token_estimate), 0) AS token_estimate,
           coalesce(sum(cost_usd), 0)       AS cost_usd,
           count(*)                         AS runs
      FROM ingest_usage
     WHERE machine_id = :machine_id
    """
)


def _require_machine(machine_id: int) -> None:
    with session_scope() as session:
        row = session.execute(
            text("SELECT id FROM machines WHERE id = :id"), {"id": machine_id}
        ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")


@router.get("/machines/{machine_id}/usage")
def machine_usage(machine_id: int) -> dict:
    _require_machine(machine_id)

    with session_scope() as session:
        totals = session.execute(_TOTALS_SQL, {"machine_id": machine_id}).mappings().one()
        ingest = session.execute(_INGEST_TOTALS_SQL, {"machine_id": machine_id}).mappings().one()
        by_model = session.execute(_BY_MODEL_SQL, {"machine_id": machine_id}).mappings().all()
        by_workflow = session.execute(_BY_WORKFLOW_SQL, {"machine_id": machine_id}).mappings().all()

    return {
        "machine_id": machine_id,
        "queries": int(totals["queries"] or 0),
        "prompt_tokens": int(totals["prompt_tokens"] or 0),
        "completion_tokens": int(totals["completion_tokens"] or 0),
        "total_tokens": int(totals["total_tokens"] or 0),
        "cost_usd": float(totals["cost_usd"] or 0),
        "ingest": {
            "runs": int(ingest["runs"] or 0),
            "char_count": int(ingest["char_count"] or 0),
            "token_estimate": int(ingest["token_estimate"] or 0),
            "cost_usd": float(ingest["cost_usd"] or 0),
            "cost_source": "estimated",
        },
        "by_model": [
            {
                "model_id": row["model_id"],
                "calls": int(row["calls"] or 0),
                "prompt_tokens": int(row["prompt_tokens"] or 0),
                "completion_tokens": int(row["completion_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
                "cost_usd": float(row["cost_usd"] or 0),
                "first_call": row["first_call"],
                "last_call": row["last_call"],
            }
            for row in by_model
        ],
        "by_workflow": [
            {
                "workflow_id": int(row["workflow_id"]),
                "workflow_name": row["workflow_name"],
                "runs": int(row["runs"] or 0),
                "avg_cost_usd": float(row["avg_cost_usd"] or 0),
                "avg_total_tokens": float(row["avg_total_tokens"] or 0),
                "avg_latency_ms": float(row["avg_latency_ms"] or 0),
                "total_cost_usd": float(row["total_cost_usd"] or 0),
                "last_run_at": row["last_run_at"],
            }
            for row in by_workflow
        ],
    }


@router.get("/usage/compare", response_model=list[WorkflowComparisonRow])
def compare_workflows(machine_id: int) -> list[WorkflowComparisonRow]:
    _require_machine(machine_id)

    with session_scope() as session:
        rows = session.execute(_BY_WORKFLOW_SQL, {"machine_id": machine_id}).mappings().all()

    return [
        WorkflowComparisonRow(
            workflow_id=int(row["workflow_id"]),
            workflow_name=row["workflow_name"],
            runs=int(row["runs"] or 0),
            avg_cost_usd=float(row["avg_cost_usd"] or 0),
            avg_total_tokens=float(row["avg_total_tokens"] or 0),
            avg_latency_ms=float(row["avg_latency_ms"] or 0),
            total_cost_usd=float(row["total_cost_usd"] or 0),
            last_run_at=row["last_run_at"],
        )
        for row in rows
    ]


@router.get("/queries/{query_id}")
def query_detail(query_id: int) -> dict:
    with session_scope() as session:
        row = session.execute(
            text(
                """
                SELECT q.*, w.name AS workflow_name
                  FROM query q
                  JOIN workflows w ON w.id = q.workflow_id
                 WHERE q.id = :id
                """
            ),
            {"id": query_id},
        ).mappings().first()

        if row is None:
            raise HTTPException(status_code=404, detail=f"Query {query_id} not found")

        calls = session.execute(
            text(
                "SELECT * FROM query_llm_call WHERE query_id = :id ORDER BY call_index"
            ),
            {"id": query_id},
        ).mappings().all()

    return {
        "query_id": int(row["id"]),
        "machine_id": int(row["machine_id"]),
        "question": row["question"],
        "answer": row["answer"],
        "workflow_name": row["workflow_name"],
        "workflow_version": int(row["workflow_version"]),
        "llm_model_id": row["llm_model_id"],
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
        "llm_calls": [
            {
                "call_index": int(call["call_index"]),
                "model_id": call["model_id"],
                "purpose": call["purpose"],
                "prompt_tokens": int(call["prompt_tokens"]),
                "completion_tokens": int(call["completion_tokens"]),
                "total_tokens": int(call["total_tokens"]),
                "cost_usd": float(call["cost_usd"]),
                "cost_source": call["cost_source"],
                "usage_raw": call["usage_raw"],
            }
            for call in calls
        ],
    }