"""Chat flow test: verify one query row, per-call usage rows, and cost.

Runs against SQLite with pgvector features stubbed, so it exercises the real query
code path without a PostgreSQL server.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from app.config import Settings
from app.llm import pricing
from app.llm import query as query_mod
from app.llm import usage as usage_mod


@pytest.fixture
def sqlite_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE machines (id INTEGER PRIMARY KEY, title TEXT)"))
        conn.execute(
            text(
                """
                CREATE TABLE workflows (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    class_key TEXT,
                    enabled INTEGER
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE query (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id INTEGER,
                    workflow_id INTEGER,
                    workflow_version INTEGER,
                    llm_model_id TEXT,
                    question TEXT,
                    answer TEXT,
                    reasoning TEXT,
                    retrieval_ms INTEGER,
                    total_ms INTEGER,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0,
                    cost_source TEXT DEFAULT 'none',
                    retrieved_chunk_ids TEXT DEFAULT '[]',
                    error TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE query_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER,
                    kind TEXT,
                    payload TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE query_llm_call (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER,
                    machine_id INTEGER,
                    model_id TEXT,
                    call_index INTEGER,
                    purpose TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    cost_source TEXT,
                    usage_raw TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO workflows (id, name, description, class_key, enabled)"
                " VALUES (1, 'embedding', 'dense', 'embedding', 1)"
            )
        )

    from app import db

    monkeypatch.setattr(db, "_engine", engine, raising=False)
    monkeypatch.setattr(db, "_session_factory", None, raising=False)

    yield engine


@pytest.fixture
def configured(monkeypatch):
    settings = Settings(
        llm_models="test/model=1.0,2.0,128000",
        embed_models="test/embed=1536,0.02",
        embed_dim=1536,
        default_llm_model="test/model",
        default_embed_model="test/embed",
        api_base="https://llm.test/v1",
        embed_api_base="https://embed.test/v1",
    )
    monkeypatch.setattr(pricing, "get_settings", lambda: settings)
    monkeypatch.setattr(query_mod, "RagWorkflow", FakeWorkflow)
    yield settings


class FakeWorkflow:
    """Stands in for RagWorkflow; emits one usage event then returns an answer."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def run(self, **run_kwargs):
        handler = usage_mod.get_usage_handler()
        handler.calls.append(
            usage_mod.PendingCall(
                model_id="test/model",
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                cost_usd=1000 * 1.0 / 1_000_000 + 500 * 2.0 / 1_000_000,
                cost_source="env",
                usage_raw={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            )
        )
        return {
            "answer": "The rated capacity is 4.5 t/h.",
            "chunks": [type("C", (), {"chunk_id": 11, "metadata": {}})()],
            "retrieval_ms": 7,
        }


@pytest.mark.asyncio
async def test_chat_records_query_and_usage(sqlite_engine, configured):
    from sqlalchemy import text as sql_text

    usage_mod.install_usage_handler()

    outcome = await query_mod.run_query(
        machine_id=1,
        workflow_id=1,
        workflow_version=1,
        workflow_class_key="embedding",
        embed_model_id="test/embed",
        llm_model_id="test/model",
        question="What is the rated capacity?",
        top_k=4,
    )

    assert outcome.answer.startswith("The rated capacity")
    assert outcome.total_tokens == 1500
    assert outcome.prompt_tokens == 1000
    assert outcome.completion_tokens == 500
    assert outcome.retrieved_chunk_ids == [11]

    # Cost = 1000*1.0/1M + 500*2.0/1M = 0.002
    assert outcome.cost_usd == pytest.approx(0.002)
    assert outcome.cost_source == "env"

    with sqlite_engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT * FROM query")).mappings().all()
        calls = conn.execute(sql_text("SELECT * FROM query_llm_call")).mappings().all()

    assert len(rows) == 1
    assert rows[0]["workflow_version"] == 1
    assert rows[0]["total_tokens"] == 1500
    assert rows[0]["answer"].startswith("The rated capacity")
    assert rows[0]["error"] is None

    assert len(calls) == 1
    assert calls[0]["purpose"] == "synthesize"
    assert calls[0]["cost_source"] == "env"
    assert calls[0]["prompt_tokens"] == 1000


@pytest.mark.asyncio
async def test_chat_failure_records_error(sqlite_engine, configured):
    class Boom(FakeWorkflow):
        async def run(self, **run_kwargs):
            raise RuntimeError("provider exploded")

    import app.llm.query as q

    q.RagWorkflow = Boom
    try:
        with pytest.raises(query_mod.QueryError):
            await query_mod.run_query(
                machine_id=1,
                workflow_id=1,
                workflow_version=1,
                workflow_class_key="embedding",
                embed_model_id="test/embed",
                llm_model_id="test/model",
                question="boom",
                top_k=4,
            )
    finally:
        q.RagWorkflow = FakeWorkflow

    from sqlalchemy import text as sql_text

    with sqlite_engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT * FROM query")).mappings().all()

    assert len(rows) == 1
    assert rows[0]["error"] is not None
    assert "provider exploded" in rows[0]["error"]
    assert rows[0]["answer"] is None


def test_usage_handler_only_records_with_active_query(monkeypatch):
    handler = usage_mod.UsageEventHandler()
    usage_mod.reset_current_query_id(usage_mod.set_current_query_id(None))
    handler.handle(_fake_event())
    assert handler.calls == []


def _fake_event():
    from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
    from llama_index.core.llms import ChatMessage, ChatResponse

    return LLMChatEndEvent(
        messages=[ChatMessage(role="user", content="hi")],
        response=ChatResponse(
            message=ChatMessage(role="assistant", content="hello"),
            raw={"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        ),
    )


def test_usage_handler_records_when_query_active(monkeypatch):
    handler = usage_mod.UsageEventHandler()
    token = usage_mod.set_current_query_id(42)
    try:
        handler.handle(_fake_event())
    finally:
        usage_mod.reset_current_query_id(token)

    assert len(handler.calls) == 1
    call = handler.calls[0]
    assert call.prompt_tokens == 5
    assert call.completion_tokens == 2
    assert call.total_tokens == 7


def test_usage_handler_reads_provider_cost(monkeypatch):
    handler = usage_mod.UsageEventHandler()
    token = usage_mod.set_current_query_id(7)
    try:
        handler.handle(_fake_event_with_cost())
    finally:
        usage_mod.reset_current_query_id(token)

    assert handler.calls[0].cost_source == "provider"
    assert handler.calls[0].cost_usd == pytest.approx(0.0009)


def _fake_event_with_cost():
    from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
    from llama_index.core.llms import ChatMessage, ChatResponse

    return LLMChatEndEvent(
        messages=[ChatMessage(role="user", content="hi")],
        response=ChatResponse(
            message=ChatMessage(role="assistant", content="hello"),
            raw={
                "model": "test/model",
                "cost": 0.0009,
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        ),
    )