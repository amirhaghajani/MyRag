"""Ingestion pipeline: documents -> chunks -> embeddings -> rows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import text

from ..db import session_scope
from ..llm import pricing
from ..retrieval.keyword import invalidate as invalidate_bm25
from .embeddings import embed_texts
from .reader import read_document, sanitize_metadata, split_documents

logger = logging.getLogger(__name__)


def set_status(document_id: int, status: str, error: str | None = None) -> None:
    with session_scope() as session:
        session.execute(
            text("UPDATE documents SET status = :status, error = :error WHERE id = :id"),
            {"status": status, "error": error, "id": document_id},
        )


def clear_chunks(document_id: int) -> None:
    with session_scope() as session:
        session.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": document_id})
        session.execute(
            text("UPDATE documents SET chunk_count = 0 WHERE id = :id"), {"id": document_id}
        )


def _set_chunk_count(document_id: int, count: int) -> None:
    with session_scope() as session:
        session.execute(
            text("UPDATE documents SET chunk_count = :count WHERE id = :id"),
            {"count": count, "id": document_id},
        )


def _machine_row(session, machine_id: int):
    return session.execute(
        text(
            """
            SELECT id, embed_model_id, embed_dim, chunk_size, chunk_overlap,
                   embed_tokens_per_char
              FROM machines
             WHERE id = :id
            """
        ),
        {"id": machine_id},
    ).mappings().first()


def _document_row(session, document_id: int):
    return session.execute(
        text("SELECT id, machine_id, filename, status FROM documents WHERE id = :id"),
        {"id": document_id},
    ).mappings().first()


def index_document(document_id: int, stored_path: Path) -> None:
    """Read, chunk, embed and persist a single document. Runs in a background task."""
    set_status(document_id, "indexing")

    try:
        document = None
        with session_scope() as session:
            document = _document_row(session, document_id)
        if document is None:
            set_status(document_id, "failed", "Document not found")
            return

        machine_id = int(document["machine_id"])

        with session_scope() as session:
            machine = _machine_row(session, machine_id)
        if machine is None:
            set_status(document_id, "failed", "Machine not found")
            return

        settings_chunks = (int(machine["chunk_size"]), int(machine["chunk_overlap"]))
        max_chunks = _max_chunks()
        embed_model_id = machine["embed_model_id"]
        ratio = (
            float(machine["embed_tokens_per_char"])
            if machine["embed_tokens_per_char"] is not None
            else None
        )

        documents = read_document(stored_path)
        chunks, truncated = split_documents(
            documents,
            chunk_size=settings_chunks[0],
            chunk_overlap=settings_chunks[1],
            max_chunks=max_chunks,
        )

        clear_chunks(document_id)
        invalidate_bm25(machine_id)

        if not chunks:
            set_status(document_id, "ready")
            _record_ingest_usage(
                machine_id=machine_id,
                document_id=document_id,
                embed_model_id=embed_model_id,
                char_count=0,
                ratio=ratio,
            )
            _set_chunk_count(document_id, 0)
            return

        texts = [node.get_content() for node in chunks]
        vectors = embed_texts(embed_model_id, texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: {len(vectors)} vectors for {len(texts)} chunks."
            )

        char_count = sum(len(t) for t in texts)
        _insert_chunks(
            document_id=document_id,
            machine_id=machine_id,
            chunks=chunks,
            vectors=vectors,
            embed_model_id=embed_model_id,
            embed_dim=int(machine["embed_dim"]),
            ratio=ratio,
        )

        cost, tokens = _record_ingest_usage(
            machine_id=machine_id,
            document_id=document_id,
            embed_model_id=embed_model_id,
            char_count=char_count,
            ratio=ratio,
        )

        _set_chunk_count(document_id, len(chunks))

        if truncated:
            _append_document_note(document_id, f"truncated at {max_chunks} chunks")

        set_status(document_id, "ready")
        logger.info(
            "Indexed document %s: %d chunks, ~%s tokens, $%.6f (%s)",
            document_id,
            len(chunks),
            tokens,
            cost,
            "estimated",
        )

    except Exception as exc:
        logger.exception("Failed to index document %s", document_id)
        set_status(document_id, "failed", str(exc)[:2000])


def _max_chunks() -> int:
    from ..config import get_settings

    return get_settings().max_chunks_per_doc


def _append_document_note(document_id: int, note: str) -> None:
    with session_scope() as session:
        session.execute(
            text("UPDATE documents SET error = :note WHERE id = :id"),
            {"note": note, "id": document_id},
        )


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def _insert_chunks(
    *,
    document_id: int,
    machine_id: int,
    chunks: list,
    vectors: list[list[float]],
    embed_model_id: str,
    embed_dim: int,
    ratio: float | None,
) -> None:
    rows = []
    for index, (node, vector) in enumerate(zip(chunks, vectors, strict=True)):
        metadata = sanitize_metadata(
            dict(node.metadata or {}), machine_id=machine_id, document_id=document_id
        )
        text_content = node.get_content()
        rows.append(
            {
                "machine_id": machine_id,
                "document_id": document_id,
                "chunk_index": index,
                "text": text_content,
                "metadata": metadata,
                "embed_model_id": embed_model_id,
                "embed_dim": embed_dim,
                "vector": vector,
                "token_estimate": pricing.estimate_embed_tokens(len(text_content), ratio),
            }
        )

    sql = text(
        """
        INSERT INTO chunks
            (machine_id, document_id, chunk_index, text, metadata,
             embed_model_id, embed_dim, vector, token_estimate)
        VALUES
            (:machine_id, :document_id, :chunk_index, :text, CAST(:metadata AS jsonb),
             :embed_model_id, :embed_dim, CAST(:vector AS vector), :token_estimate)
        """
    )
    with session_scope() as session:
        # Pass JSON text and let the CAST do the work: psycopg3 cannot adapt a
        # bare dict, and SQLite has no jsonb at all.
        for row in rows:
            bind = dict(row)
            bind["metadata"] = json.dumps(row["metadata"])
            bind["vector"] = _vector_literal(row["vector"])
            session.execute(sql, bind)


def _record_ingest_usage(
    *,
    machine_id: int,
    document_id: int,
    embed_model_id: str,
    char_count: int,
    ratio: float | None,
) -> tuple[float, int]:
    cost_result, tokens = pricing.resolve_embed_cost(
        model_id=embed_model_id, char_count=char_count, ratio=ratio
    )

    with session_scope() as session:
        session.execute(
            text(
                """
                INSERT INTO ingest_usage
                    (machine_id, document_id, embed_model_id, char_count,
                     token_estimate, cost_usd, cost_source)
                VALUES
                    (:machine_id, :document_id, :embed_model_id, :char_count,
                     :token_estimate, :cost_usd, :cost_source)
                """
            ),
            {
                "machine_id": machine_id,
                "document_id": document_id,
                "embed_model_id": embed_model_id,
                "char_count": char_count,
                "token_estimate": tokens,
                "cost_usd": cost_result.usd,
                "cost_source": cost_result.source,
            },
        )

    return cost_result.usd, tokens