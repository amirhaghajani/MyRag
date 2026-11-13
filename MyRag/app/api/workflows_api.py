"""Workflow listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from ..db import session_scope
from ..schemas import WorkflowOut

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


@router.get("", response_model=list[WorkflowOut])
def list_workflows() -> list[WorkflowOut]:
    with session_scope() as session:
        rows = session.execute(text("SELECT * FROM workflows ORDER BY id")).mappings().all()

    return [
        WorkflowOut(
            id=int(row["id"]),
            name=row["name"],
            description=row["description"],
            class_key=row["class_key"],
            enabled=bool(row["enabled"]),
        )
        for row in rows
    ]