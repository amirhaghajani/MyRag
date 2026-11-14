"""Document upload, indexing, listing and deletion."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from sqlalchemy import text

from ..config import get_settings
from ..db import session_scope
from ..ingest.pipeline import index_document, set_status
from ..ingest.reader import UnsupportedFileError, validate_upload
from ..schemas import DocumentOut

router = APIRouter(tags=["documents"])

# Overridable so tests can point uploads at a temporary directory.
UPLOAD_ROOT = Path(os.environ.get("MYRAG_UPLOAD_DIR", "data/uploads"))

_LIST_SQL = text(
    """
    SELECT d.*,
           (SELECT coalesce(sum(iu.cost_usd), 0)
              FROM ingest_usage iu WHERE iu.document_id = d.id) AS embed_cost_usd,
           (SELECT coalesce(sum(iu.token_estimate), 0)
              FROM ingest_usage iu WHERE iu.document_id = d.id) AS token_estimate
      FROM documents d
     WHERE d.machine_id = :machine_id
     ORDER BY d.id DESC
    """
)


def _to_out(row) -> DocumentOut:
    return DocumentOut(
        id=int(row["id"]),
        machine_id=int(row["machine_id"]),
        filename=row["filename"],
        content_type=row["content_type"],
        byte_size=int(row["byte_size"]),
        status=row["status"],
        error=row["error"],
        created_at=row["created_at"],
        chunk_count=int(row["chunk_count"] or 0),
        embed_cost_usd=float(row["embed_cost_usd"] or 0),
        token_estimate=int(row["token_estimate"] or 0),
    )


def _ensure_embed_compatible(machine) -> None:
    """Block uploads/indexing when the machine's embed model is out of date."""
    settings = get_settings()
    embed_dim = int(machine["embed_dim"])
    model_id = machine["embed_model_id"]

    if embed_dim != settings.embed_dim:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This machine embeds with dim {embed_dim}, but the server is configured "
                f"for dim {settings.embed_dim}. Pick a matching embed model in machine "
                "settings and re-embed its documents before uploading more."
            ),
        )
    if model_id not in settings.embed_prices:
        known = ", ".join(sorted(settings.embed_prices))
        raise HTTPException(
            status_code=409,
            detail=(
                f"Embed model '{model_id}' is no longer configured. Configured: {known}. "
                "Change the machine's embed model and re-embed its documents."
            ),
        )


@router.post("/api/machines/{machine_id}/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    machine_id: int,
    file: UploadFile = File(...),
) -> DocumentOut:
    """Store an uploaded file. Indexing is a separate call so the client can
    re-index without re-uploading."""
    with session_scope() as session:
        machine = session.execute(
            text("SELECT id, embed_model_id, embed_dim FROM machines WHERE id = :id"),
            {"id": machine_id},
        ).mappings().first()
    if machine is None:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    _ensure_embed_compatible(machine)

    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=422, detail="Uploaded file has no name.")

    machine_dir = UPLOAD_ROOT / str(machine_id)
    machine_dir.mkdir(parents=True, exist_ok=True)
    stored_path = machine_dir / f"{uuid.uuid4().hex}{Path(filename).suffix.lower()}"

    byte_size = 0
    try:
        with stored_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                byte_size += len(chunk)
                if byte_size > 1024 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File exceeds the 1 GB hard limit.")
                handle.write(chunk)

        validate_upload(filename, file.content_type or "", byte_size)
    except (UnsupportedFileError, ValueError) as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise

    with session_scope() as session:
        result = session.execute(
            text(
                """
                INSERT INTO documents
                    (machine_id, filename, content_type, byte_size, stored_path, status)
                VALUES
                    (:machine_id, :filename, :content_type, :byte_size, :stored_path, 'uploaded')
                RETURNING *
                """
            ),
            {
                "machine_id": machine_id,
                "filename": filename,
                "content_type": file.content_type or "application/octet-stream",
                "byte_size": byte_size,
                "stored_path": str(stored_path),
            },
        )
        row = result.mappings().one()

    return _to_out({**dict(row), "embed_cost_usd": 0.0, "token_estimate": 0})


@router.post("/api/documents/{document_id}/index", status_code=202)
def start_indexing(document_id: int, background_tasks: BackgroundTasks) -> dict:
    with session_scope() as session:
        row = session.execute(
            text(
                """
                SELECT d.id, d.machine_id, d.filename, d.status, d.stored_path
                  FROM documents d
                 WHERE d.id = :id
                """
            ),
            {"id": document_id},
        ).mappings().first()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    with session_scope() as session:
        machine = session.execute(
            text("SELECT embed_model_id, embed_dim FROM machines WHERE id = :id"),
            {"id": int(row["machine_id"])},
        ).mappings().first()
    if machine is None:
        raise HTTPException(status_code=404, detail=f"Machine {row['machine_id']} not found")
    _ensure_embed_compatible(machine)

    if row["status"] == "indexing":
        raise HTTPException(status_code=409, detail="Document is already being indexed.")
    if not row["stored_path"]:
        set_status(document_id, "failed", "No stored file path recorded for this document.")
        raise HTTPException(status_code=410, detail="Stored file path is missing; re-upload.")

    stored_path = Path(row["stored_path"])
    if not stored_path.exists():
        set_status(document_id, "failed", f"Stored file missing at {stored_path}")
        raise HTTPException(status_code=410, detail="Stored file is missing; re-upload.")

    background_tasks.add_task(index_document, document_id, stored_path)
    return {"document_id": document_id, "status": "indexing"}


@router.get("/api/machines/{machine_id}/documents", response_model=list[DocumentOut])
def list_documents(machine_id: int) -> list[DocumentOut]:
    with session_scope() as session:
        rows = session.execute(_LIST_SQL, {"machine_id": machine_id}).mappings().all()
    return [_to_out(row) for row in rows]


@router.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: int) -> None:
    from ..retrieval.keyword import invalidate as invalidate_bm25

    with session_scope() as session:
        row = session.execute(
            text("SELECT id, machine_id, stored_path FROM documents WHERE id = :id"),
            {"id": document_id},
        ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        machine_id = int(row["machine_id"])
        session.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": document_id})
        session.execute(
            text("UPDATE documents SET chunk_count = 0 WHERE id = :id"), {"id": document_id}
        )
        session.execute(
            text("DELETE FROM ingest_usage WHERE document_id = :id"), {"id": document_id}
        )
        session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})

    invalidate_bm25(machine_id)

    if row["stored_path"]:
        Path(row["stored_path"]).unlink(missing_ok=True)