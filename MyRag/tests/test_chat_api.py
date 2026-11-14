"""End-to-end chat through the HTTP API with a stubbed LLM.

Verifies the whole path: request -> workflow -> workflow row -> query row ->
per-call usage -> cost -> response payload. No network or LLM calls are made.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

import app.db as db
from app.config import Settings
from tests.fakes import FakeAsyncClient, FakeClient
from tests.sqlite_schema import _ddl_statements


@pytest.fixture
def client(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Keep uploaded files out of the repo.
    import app.api.documents as documents_mod

    monkeypatch.setattr(documents_mod, "UPLOAD_ROOT", tmp_path / "uploads", raising=False)

    with engine.begin() as conn:
        for statement in _ddl_statements():
            conn.execute(text(statement))

    monkeypatch.setattr(db, "_engine", engine, raising=False)
    monkeypatch.setattr(db, "_session_factory", None, raising=False)
    monkeypatch.setattr(db, "create_schema", lambda *args, **kwargs: True, raising=False)

    import app.ingest.pipeline as pipeline_mod
    import app.retrieval.embedding as embedding_mod

    fake_embed = lambda model_id, texts: [[0.1] * 8 for _ in texts]  # noqa: E731
    monkeypatch.setattr(pipeline_mod, "embed_texts", fake_embed)
    monkeypatch.setattr(embedding_mod, "embed_texts", fake_embed)

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

    import app.config as config_mod

    monkeypatch.setattr(config_mod, "get_settings", lambda: settings, raising=False)
    for module_name in (
        "app.api.models_api",
        "app.api.machines",
        "app.api.documents",
        "app.api.chat",
        "app.ingest.reader",
        "app.llm.pricing",
        "app.llm.factory",
        "app.llm.query",
        "app.retrieval.embedding",
    ):
        module = __import__(module_name, fromlist=["get_settings"])
        if hasattr(module, "get_settings"):
            monkeypatch.setattr(module, "get_settings", lambda: settings, raising=False)

    # Stub the OpenAI clients so llama_index's own instrumentation fires and the
    # usage handler sees real events (a fake LLM object would bypass them).
    # _get_client/_get_aclient live on the OpenAI base class, so patch them there.
    from llama_index.llms.openai.base import OpenAI as OpenAIBase

    monkeypatch.setattr(OpenAIBase, "_get_aclient", lambda self: FakeAsyncClient(), raising=False)
    monkeypatch.setattr(OpenAIBase, "_get_client", lambda self: FakeClient(), raising=False)

    import app.main as main_mod

    with TestClient(main_mod.app) as test_client:
        test_client.engine = engine
        yield test_client


def _make_machine_with_doc(client):
    machine_id = client.post("/api/machines", json={"title": "Mill"}).json()["id"]
    files = {
        "file": ("motor.txt", io.BytesIO(b"Rated capacity: 4.5 t/h\nMotor: 45 kW"), "text/plain")
    }
    doc_id = client.post(f"/api/machines/{machine_id}/documents", files=files).json()["id"]
    client.post(f"/api/documents/{doc_id}/index")
    return machine_id, doc_id


def _keyword_workflow_id(client):
    """The keyword flow is pure Python, so it runs on SQLite (no pgvector ops)."""
    for row in client.get("/api/workflows").json():
        if row["name"] == "keyword":
            return row["id"]
    raise AssertionError("keyword workflow not seeded")


def _last_error(client) -> str | None:
    """Read the stored error from the most recent query row, for clearer failures."""
    with client.engine.begin() as conn:
        return conn.execute(text("SELECT error FROM query ORDER BY id DESC LIMIT 1")).scalar()


def test_chat_returns_answer_and_cost(client):
    machine_id, _ = _make_machine_with_doc(client)
    workflow_id = _keyword_workflow_id(client)

    response = client.post(
        f"/api/machines/{machine_id}/chat",
        json={
            "question": "What is the rated capacity?",
            "workflow_id": workflow_id,
            "llm_model_id": "test/model",
        },
    )

    assert response.status_code == 200, f"{response.text} | db error: {_last_error(client)}"
    body = response.json()
    assert body["answer"].startswith("The rated capacity")
    assert body["total_tokens"] == 420
    assert body["prompt_tokens"] == 300
    assert body["completion_tokens"] == 120
    # cost = 300*1.0/1M + 120*2.0/1M
    assert body["cost_usd"] == pytest.approx(0.00054)
    assert body["cost_source"] == "env"
    assert body["workflow_name"] == "keyword"
    assert len(body["llm_calls"]) == 1
    assert body["llm_calls"][0]["purpose"] == "synthesize"


def test_query_log_returns_retrieval_and_prompt(client):
    machine_id, _ = _make_machine_with_doc(client)
    workflow_id = _keyword_workflow_id(client)

    body = client.post(
        f"/api/machines/{machine_id}/chat",
        json={
            "question": "What is the rated capacity?",
            "workflow_id": workflow_id,
            "llm_model_id": "test/model",
        },
    ).json()

    response = client.get(f"/api/queries/{body['query_id']}/log")
    assert response.status_code == 200, response.text
    events = response.json()["events"]
    kinds = [event["kind"] for event in events]
    assert kinds == ["retrieval", "prompt"]

    retrieval = events[0]["payload"]
    assert retrieval["chunks"]
    assert retrieval["chunks"][0]["text"]
    assert "filename" in retrieval["chunks"][0]

    prompt = events[1]["payload"]
    assert prompt["messages"][0]["role"] == "system"
    assert "Documentation:" in prompt["messages"][-1]["content"]


def test_query_log_missing_query_returns_404(client):
    assert client.get("/api/queries/999999/log").status_code == 404


def test_chat_blocked_when_machine_embed_is_stale(client):
    machine_id, _ = _make_machine_with_doc(client)
    workflow_id = _keyword_workflow_id(client)
    with db.get_engine().begin() as conn:
        conn.execute(
            text("UPDATE machines SET embed_model_id = 'gone/model' WHERE id = :id"),
            {"id": machine_id},
        )

    response = client.post(
        f"/api/machines/{machine_id}/chat",
        json={
            "question": "Rated capacity?",
            "workflow_id": workflow_id,
            "llm_model_id": "test/model",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "embed_model_stale"


def test_chat_persists_query_and_records_version(client):
    machine_id, _ = _make_machine_with_doc(client)
    workflow_id = _keyword_workflow_id(client)

    created = client.post(
        f"/api/machines/{machine_id}/chat",
        json={
            "question": "Rated capacity?",
            "workflow_id": workflow_id,
            "llm_model_id": "test/model",
        },
    ).json()

    messages = client.get(f"/api/machines/{machine_id}/messages").json()
    assert len(messages) == 1
    assert messages[0]["query_id"] == created["query_id"]
    assert messages[0]["workflow_version"] == workflow_id
    assert messages[0]["cost_usd"] == pytest.approx(0.00054)

    detail = client.get(f"/api/queries/{created['query_id']}").json()
    assert detail["llm_calls"][0]["prompt_tokens"] == 300


def test_usage_totals_reflect_chat(client):
    machine_id, _ = _make_machine_with_doc(client)
    workflow_id = _keyword_workflow_id(client)

    client.post(
        f"/api/machines/{machine_id}/chat",
        json={
            "question": "Capacity?",
            "workflow_id": workflow_id,
            "llm_model_id": "test/model",
        },
    )

    usage = client.get(f"/api/machines/{machine_id}/usage").json()
    assert usage["queries"] == 1
    assert usage["total_tokens"] == 420
    assert usage["cost_usd"] == pytest.approx(0.00054)
    assert usage["ingest"]["cost_source"] == "estimated"

    compare = client.get(f"/api/usage/compare?machine_id={machine_id}").json()
    assert len(compare) == 1
    assert compare[0]["workflow_name"] == "keyword"
    assert compare[0]["runs"] == 1
    assert compare[0]["avg_cost_usd"] == pytest.approx(0.00054)


def test_chat_rejects_unknown_model(client):
    machine_id, _ = _make_machine_with_doc(client)
    workflow_id = client.get("/api/workflows").json()[0]["id"]

    response = client.post(
        f"/api/machines/{machine_id}/chat",
        json={
            "question": "q",
            "workflow_id": workflow_id,
            "llm_model_id": "nope/model",
        },
    )
    assert response.status_code == 422


def test_chat_rejects_unknown_workflow(client):
    machine_id, _ = _make_machine_with_doc(client)

    response = client.post(
        f"/api/machines/{machine_id}/chat",
        json={
            "question": "q",
            "workflow_id": 9999,
            "llm_model_id": "test/model",
        },
    )
    assert response.status_code == 404


def test_chat_for_missing_machine_returns_404(client):
    workflow_id = client.get("/api/workflows").json()[0]["id"]
    response = client.post(
        "/api/machines/999/chat",
        json={"question": "q", "workflow_id": workflow_id, "llm_model_id": "test/model"},
    )
    assert response.status_code == 404
