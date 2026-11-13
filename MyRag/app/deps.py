"""Shared query helpers for the API layer."""

from __future__ import annotations

from sqlalchemy import text

from .db import session_scope


def fetch_machine(session, machine_id: int):
    return session.execute(
        text("SELECT * FROM machines WHERE id = :id"), {"id": machine_id}
    ).mappings().first()


def fetch_workflow(session, workflow_id: int):
    return session.execute(
        text("SELECT * FROM workflows WHERE id = :id"), {"id": workflow_id}
    ).mappings().first()


def fetch_workflow_by_key(session, class_key: str):
    return session.execute(
        text("SELECT * FROM workflows WHERE class_key = :key"), {"key": class_key}
    ).mappings().first()


def machine_guard(machine_id: int) -> dict:
    """Return the machine row or raise MachineNotFound."""
    with session_scope() as session:
        row = fetch_machine(session, machine_id)
    if row is None:
        raise MachineNotFound(machine_id)
    return dict(row)


class MachineNotFound(LookupError):
    def __init__(self, machine_id: int) -> None:
        super().__init__(f"Machine {machine_id} not found")
        self.machine_id = machine_id


class WorkflowNotFound(LookupError):
    def __init__(self, workflow_id: int) -> None:
        super().__init__(f"Workflow {workflow_id} not found")
        self.workflow_id = workflow_id


def workflow_guard(workflow_id: int) -> dict:
    with session_scope() as session:
        row = fetch_workflow(session, workflow_id)
    if row is None:
        raise WorkflowNotFound(workflow_id)
    if not row["enabled"]:
        raise LookupError(f"Workflow '{row['name']}' is disabled")
    return dict(row)