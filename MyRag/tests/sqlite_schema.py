"""SQLite schema for tests.

Postgres uses pgvector types that SQLite cannot represent, so tests build a plain
SQL version of the same tables. This is the single source of that DDL.
"""

from __future__ import annotations

TABLES = [
    """
    CREATE TABLE machines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        embed_model_id TEXT NOT NULL,
        embed_dim INTEGER NOT NULL,
        chunk_size INTEGER NOT NULL,
        chunk_overlap INTEGER NOT NULL,
        embed_tokens_per_char NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        content_type TEXT NOT NULL,
        byte_size INTEGER NOT NULL,
        stored_path TEXT,
        status TEXT DEFAULT 'uploaded',
        error TEXT,
        chunk_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id INTEGER NOT NULL,
        document_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        metadata TEXT DEFAULT '{}',
        embed_model_id TEXT,
        embed_dim INTEGER,
        vector TEXT,
        vector_8 TEXT,
        token_estimate INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE workflows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL,
        class_key TEXT NOT NULL,
        enabled INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
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
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
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
        usage_raw TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE query_event (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id INTEGER,
        kind TEXT,
        payload TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE ingest_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id INTEGER,
        document_id INTEGER,
        embed_model_id TEXT,
        char_count INTEGER,
        token_estimate INTEGER,
        cost_usd REAL,
        cost_source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


def _ddl_statements() -> list[str]:
    return list(TABLES)