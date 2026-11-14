"""Machine CRUD endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import text

from ..config import get_settings
from ..db import session_scope
from ..deps import MachineNotFound, fetch_machine
from ..schemas import MachineCreate, MachineOut, MachineSummary, MachineUpdate

router = APIRouter(prefix="/api/machines", tags=["machines"])

_SUMMARY_SQL_TEXT = """
    SELECT m.*,
           (SELECT count(*) FROM documents d WHERE d.machine_id = m.id) AS document_count,
           (SELECT count(*) FROM chunks c WHERE c.machine_id = m.id) AS chunk_count,
           (SELECT count(*) FROM query q WHERE q.machine_id = m.id) AS query_count,
           (SELECT coalesce(sum(q.cost_usd), 0) FROM query q WHERE q.machine_id = m.id)
                + (SELECT coalesce(sum(iu.cost_usd), 0)
                     FROM ingest_usage iu WHERE iu.machine_id = m.id)
           AS total_cost_usd
      FROM machines m
    """

_SUMMARY_SQL = text(_SUMMARY_SQL_TEXT)
_SUMMARY_BY_ID_SQL = text(_SUMMARY_SQL_TEXT + " WHERE m.id = :id")


def _row_to_out(row) -> MachineOut:
    return MachineOut(
        id=int(row["id"]),
        title=row["title"],
        embed_model_id=row["embed_model_id"],
        embed_dim=int(row["embed_dim"]),
        chunk_size=int(row["chunk_size"]),
        chunk_overlap=int(row["chunk_overlap"]),
        created_at=row["created_at"],
    )


@router.post("", response_model=MachineOut, status_code=201)
def create_machine(payload: MachineCreate) -> MachineOut:
    settings = get_settings()
    embed_model_id = payload.embed_model_id or settings.default_embed_model
    embed_price = settings.embed_prices.get(embed_model_id)

    if embed_price is None:
        known = ", ".join(sorted(settings.embed_prices))
        raise HTTPException(
            status_code=422,
            detail=f"Unknown embed model '{embed_model_id}'. Configured: {known}",
        )
    if embed_price.dim != settings.embed_dim:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Embed model '{embed_model_id}' has dim {embed_price.dim} but EMBED_DIM is "
                f"{settings.embed_dim}. All configured embed models must match EMBED_DIM."
            ),
        )

    chunk_size = payload.chunk_size or settings.chunk_size
    chunk_overlap = (
        payload.chunk_overlap if payload.chunk_overlap is not None else settings.chunk_overlap
    )
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=422, detail="chunk_overlap must be smaller than chunk_size."
        )

    with session_scope() as session:
        result = session.execute(
            text(
                """
                INSERT INTO machines
                    (title, embed_model_id, embed_dim, chunk_size, chunk_overlap)
                VALUES
                    (:title, :embed_model_id, :embed_dim, :chunk_size, :chunk_overlap)
                RETURNING *
                """
            ),
            {
                "title": payload.title.strip(),
                "embed_model_id": embed_model_id,
                "embed_dim": settings.embed_dim,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        )
        row = result.mappings().one()

    return _row_to_out(row)


@router.get("", response_model=list[MachineSummary])
def list_machines() -> list[MachineSummary]:
    with session_scope() as session:
        rows = session.execute(_SUMMARY_SQL).mappings().all()

    summaries = []
    for row in rows:
        base = _row_to_out(row).model_dump()
        summaries.append(
            MachineSummary(
                **base,
                document_count=int(row["document_count"] or 0),
                chunk_count=int(row["chunk_count"] or 0),
                query_count=int(row["query_count"] or 0),
                total_cost_usd=float(row["total_cost_usd"] or 0),
            )
        )
    return summaries


@router.get("/{machine_id}", response_model=MachineSummary)
def get_machine(machine_id: int) -> MachineSummary:
    with session_scope() as session:
        row = session.execute(
            _SUMMARY_BY_ID_SQL, {"id": machine_id}
        ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")

    return MachineSummary(
        **_row_to_out(row).model_dump(),
        document_count=int(row["document_count"] or 0),
        chunk_count=int(row["chunk_count"] or 0),
        query_count=int(row["query_count"] or 0),
        total_cost_usd=float(row["total_cost_usd"] or 0),
    )


@router.patch("/{machine_id}", response_model=MachineSummary, status_code=202)
def update_machine(
    machine_id: int, payload: MachineUpdate, background_tasks: BackgroundTasks
) -> MachineSummary:
    """Switch a machine to another embed model.

    Re-embedding is required whenever the machine already has chunks: the stored
    vectors were produced by the old model. The client must confirm that intent with
    ``reembed: true``.
    """
    from ..ingest.pipeline import index_document

    settings = get_settings()
    model_id = payload.embed_model_id

    if model_id is not None:
        price = settings.embed_prices.get(model_id)
        if price is None:
            known = ", ".join(sorted(settings.embed_prices))
            raise HTTPException(
                status_code=422,
                detail=f"Unknown embed model '{model_id}'. Configured: {known}",
            )
        if price.dim != settings.embed_dim:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Embed model '{model_id}' has dim {price.dim} but EMBED_DIM is "
                    f"{settings.embed_dim}. A dimension change needs a full re-embed: run "
                    "'python -m app.admin reembed --confirm --embed-model "
                    f"{model_id}' with the server stopped."
                ),
            )

    with session_scope() as session:
        row = session.execute(
            text(
                "SELECT id, embed_model_id, embed_dim FROM machines WHERE id = :id"
            ),
            {"id": machine_id},
        ).mappings().first()
        if row is None:
            raise MachineNotFound(machine_id)

        documents = session.execute(
            text(
                "SELECT id, stored_path FROM documents WHERE machine_id = :id ORDER BY id"
            ),
            {"id": machine_id},
        ).mappings().all()

        chunk_count = int(
            session.execute(
                text("SELECT count(*) FROM chunks WHERE machine_id = :id"),
                {"id": machine_id},
            ).scalar_one()
        )

        if chunk_count > 0 and not payload.reembed:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Machine {machine_id} has {chunk_count} chunks. Changing the embed "
                    "model re-embeds them (embedding API cost). Resend with "
                    '"reembed": true to confirm.'
                ),
            )

        if model_id is not None:
            session.execute(
                text(
                    "UPDATE machines SET embed_model_id = :model, embed_dim = :dim "
                    "WHERE id = :id"
                ),
                {"model": model_id, "dim": settings.embed_dim, "id": machine_id},
            )

    if payload.reembed:
        for document in documents:
            stored = document["stored_path"]
            if stored and Path(stored).exists():
                background_tasks.add_task(index_document, int(document["id"]), Path(stored))
            else:
                from ..ingest.pipeline import set_status

                set_status(int(document["id"]), "failed", "Stored file missing; re-upload.")

    with session_scope() as session:
        row = session.execute(
            _SUMMARY_BY_ID_SQL, {"id": machine_id}
        ).mappings().one()

    return MachineSummary(
        **_row_to_out(row).model_dump(),
        document_count=int(row["document_count"] or 0),
        chunk_count=int(row["chunk_count"] or 0),
        query_count=int(row["query_count"] or 0),
        total_cost_usd=float(row["total_cost_usd"] or 0),
    )


@router.delete("/{machine_id}", status_code=204)
def delete_machine(machine_id: int) -> None:
    with session_scope() as session:
        row = fetch_machine(session, machine_id)
        if row is None:
            raise MachineNotFound(machine_id)

        session.execute(text("DELETE FROM documents WHERE machine_id = :id"), {"id": machine_id})
        session.execute(text("DELETE FROM chunks WHERE machine_id = :id"), {"id": machine_id})
        session.execute(
            text("DELETE FROM query_llm_call WHERE machine_id = :id"), {"id": machine_id}
        )
        session.execute(text("DELETE FROM query WHERE machine_id = :id"), {"id": machine_id})
        session.execute(text("DELETE FROM ingest_usage WHERE machine_id = :id"), {"id": machine_id})
        session.execute(text("DELETE FROM machines WHERE id = :id"), {"id": machine_id})

    from ..retrieval.keyword import invalidate as invalidate_bm25

    invalidate_bm25(machine_id)
