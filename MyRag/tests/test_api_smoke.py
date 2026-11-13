"""End-to-end API smoke test against SQLite.

Exercises the real HTTP surface (machines, documents, workflows, models) without
requiring PostgreSQL. Vector search is skipped because SQLite has no pgvector.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

import app.db as db
import app.ingest.embeddings as embeddings_mod
from app.config import Settings
from tests.sqlite_schema import _ddl_statements


@pytest.fixture
def client(monkeypatch, tmp_path):
    # StaticPool keeps one connection so every session sees the same in-memory DB.
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Keep uploaded files out of the repo.
    import app.api.documents as documents_mod

    monkeypatch.setattr(documents_mod, "UPLOAD_ROOT", tmp_path / "uploads", raising=False)

    # Build tables with plain SQL (pgvector types are unavailable on SQLite).
    with engine.begin() as conn:
        for statement in _ddl_statements():
            conn.execute(text(statement))

    monkeypatch.setattr(db, "_engine", engine, raising=False)
    monkeypatch.setattr(db, "_session_factory", None, raising=False)
    monkeypatch.setattr(db, "create_schema", lambda dim: True, raising=False)

    # Stub embeddings so ingestion needs no network. Patch the source module so
    # the pipeline's already-imported reference sees the stub too.
    monkeypatch.setattr(
        embeddings_mod, "embed_texts", lambda model_id, texts: [[0.1] * 8 for _ in texts]
    )
    monkeypatch.setattr(embeddings_mod, "check_embed_model", lambda model_id: 8, raising=False)

    import app.ingest.pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod, "embed_texts", lambda model_id, texts: [[0.1] * 8 for _ in texts]
    )

    settings = Settings(
        llm_models="test/model=1.0,2.0,128000",
        embed_models="test/embed=8,0.02",
        embed_dim=8,
        default_llm_model="test/model",
        default_embed_model="test/embed",
        api_base="https://llm.test/v1",
        embed_api_base="https://embed.test/v1",
        chunk_size=128,
        chunk_overlap=16,
    )

    # Settings are cached process-wide; patching the cache reaches every router.
    import app.config as config_mod

    monkeypatch.setattr(config_mod, "_cached_settings", settings, raising=False)
    monkeypatch.setattr(config_mod, "get_settings", lambda: settings, raising=False)
    modules = (
        "app.api.models_api",
        "app.api.machines",
        "app.api.chat",
        "app.ingest.reader",
        "app.llm.pricing",
    )
    for module_name in modules:
        module = __import__(module_name, fromlist=["get_settings"])
        if hasattr(module, "get_settings"):
            monkeypatch.setattr(module, "get_settings", lambda: settings, raising=False)

    import app.main as main_mod

    with TestClient(main_mod.app) as client:
        yield client


def _seed_workflow(client):
    return client.get("/api/workflows").json()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint_reports_both_endpoints(client):
    body = client.get("/api/models").json()
    assert body["llm_api_base"] == "https://llm.test/v1"
    assert body["embed_api_base"] == "https://embed.test/v1"
    assert [m["id"] for m in body["llms"]] == ["test/model"]
    assert [m["id"] for m in body["embeddings"]] == ["test/embed"]
    assert body["embed_dim"] == 8


def test_workflows_are_seeded(client):
    rows = _seed_workflow(client)
    assert {r["name"] for r in rows} == {"embedding", "keyword", "hybrid"}
    assert all(r["class_key"] == r["name"] for r in rows)


def test_create_and_list_machine(client):
    created = client.post("/api/machines", json={"title": "Roller Mill #2"})
    assert created.status_code == 201
    machine = created.json()
    assert machine["title"] == "Roller Mill #2"
    assert machine["embed_model_id"] == "test/embed"

    listed = client.get("/api/machines").json()
    assert len(listed) == 1
    assert listed[0]["document_count"] == 0
    assert listed[0]["total_cost_usd"] == 0.0


def test_create_machine_rejects_unknown_embed_model(client):
    response = client.post(
        "/api/machines", json={"title": "X", "embed_model_id": "nope/model"}
    )
    assert response.status_code == 422


def test_create_machine_rejects_bad_overlap(client):
    response = client.post(
        "/api/machines", json={"title": "X", "chunk_size": 100, "chunk_overlap": 200}
    )
    assert response.status_code == 422


def test_upload_and_index_txt(client):
    machine_id = client.post("/api/machines", json={"title": "Mill"}).json()["id"]

    files = {
        "file": ("motor.txt", io.BytesIO(b"Rated capacity: 4.5 t/h\nMotor: 45 kW"), "text/plain")
    }
    upload = client.post(f"/api/machines/{machine_id}/documents", files=files)
    assert upload.status_code == 201
    doc = upload.json()
    assert doc["filename"] == "motor.txt"
    assert doc["status"] == "uploaded"

    index = client.post(f"/api/documents/{doc['id']}/index")
    assert index.status_code == 202

    # Background tasks run before the TestClient request returns.
    docs = client.get(f"/api/machines/{machine_id}/documents").json()
    indexed = [d for d in docs if d["id"] == doc["id"]][0]
    assert indexed["status"] == "ready"
    assert indexed["chunk_count"] > 0
    assert indexed["token_estimate"] > 0
    assert indexed["embed_cost_usd"] > 0


def test_upload_rejects_unsupported_type(client):
    machine_id = client.post("/api/machines", json={"title": "Mill"}).json()["id"]
    files = {"file": ("note.md", io.BytesIO(b"# hi"), "text/markdown")}
    response = client.post(f"/api/machines/{machine_id}/documents", files=files)
    assert response.status_code == 422


def test_index_missing_document_returns_404(client):
    response = client.post("/api/documents/999/index")
    assert response.status_code == 404


def test_usage_is_empty_before_chat(client):
    machine_id = client.post("/api/machines", json={"title": "Mill"}).json()["id"]
    body = client.get(f"/api/machines/{machine_id}/usage").json()
    assert body["queries"] == 0
    assert body["cost_usd"] == 0.0
    assert body["ingest"]["cost_source"] == "estimated"
    assert client.get(f"/api/usage/compare?machine_id={machine_id}").json() == []


def test_usage_for_missing_machine_returns_404(client):
    assert client.get("/api/machines/999/usage").status_code == 404


def test_delete_machine_cascades(client):
    machine_id = client.post("/api/machines", json={"title": "Mill"}).json()["id"]
    files = {"file": ("a.txt", io.BytesIO(b"hello world"), "text/plain")}
    doc_id = client.post(f"/api/machines/{machine_id}/documents", files=files).json()["id"]
    client.post(f"/api/documents/{doc_id}/index")

    assert client.delete(f"/api/machines/{machine_id}").status_code == 204
    assert client.get("/api/machines").json() == []
    assert client.get(f"/api/machines/{machine_id}/usage").status_code == 404
