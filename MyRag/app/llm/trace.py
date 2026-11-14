"""Per-query trace events: retrieved chunks and the exact LLM prompt.

Rows are written while a query runs and read back by the UI's "Show log" action.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import text

from ..db import session_scope
from .usage import get_current_query_id

logger = logging.getLogger(__name__)


def record(kind: str, payload: dict) -> None:
    """Attach an event to the query currently being executed, if any."""
    query_id = get_current_query_id()
    if query_id is None:
        return
    try:
        with session_scope() as session:
            session.execute(
                text(
                    "INSERT INTO query_event (query_id, kind, payload) "
                    "VALUES (:query_id, :kind, :payload)"
                ),
                {
                    "query_id": query_id,
                    "kind": kind,
                    "payload": json.dumps(payload, ensure_ascii=False),
                },
            )
    except Exception:
        logger.warning("Could not record %s trace for query %s", kind, query_id, exc_info=True)


def record_retrieval(chunks: list) -> None:
    record(
        "retrieval",
        {
            "chunks": [
                {
                    "chunk_id": int(chunk.chunk_id),
                    "document_id": int(chunk.document_id),
                    "filename": (chunk.metadata or {}).get("filename"),
                    "score": round(float(chunk.score), 6),
                    "rank": int(chunk.rank),
                    "text": chunk.text,
                }
                for chunk in chunks
            ]
        },
    )


def record_prompt(messages: list) -> None:
    record(
        "prompt",
        {
            "messages": [
                {
                    "role": getattr(message.role, "value", str(message.role)),
                    "content": message.content or "",
                }
                for message in messages
            ]
        },
    )