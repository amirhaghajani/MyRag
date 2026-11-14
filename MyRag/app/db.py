"""Database engine, session handling and schema bootstrap."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            future=True,
            # Fail fast instead of hanging when Postgres is unreachable.
            connect_args={"connect_timeout": 5},
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session that is always closed."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def pgvector_available(conn) -> bool:
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return True
    except Exception as exc:  # noqa: BLE001 - any DB error means the extension is unusable
        logger.warning("pgvector extension unavailable: %s", exc)
        return False


def _column_type(dialect_name: str, embed_dim: int):
    if dialect_name != "postgresql":
        raise RuntimeError(
            f"MyRag requires PostgreSQL with pgvector; got dialect '{dialect_name}'."
        )
    from pgvector.sqlalchemy import Vector

    return Vector(embed_dim)


def vector_dim(conn) -> int | None:
    """Dimensions of the chunks.vector column, or None when the table is absent."""
    row = conn.execute(
        text(
            "SELECT a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = 'chunks' AND a.attname = 'vector' AND a.attnum > 0"
        )
    ).first()
    return int(row[0]) if row is not None else None


def set_vector_dim(conn, embed_dim: int) -> None:
    """Change the shared vector column. Callers must clear chunks first: the column
    is a single fixed size and Postgres cannot cast a populated 1024-dim column to
    4069 dims."""
    conn.execute(text("DROP INDEX IF EXISTS chunks_vector_hnsw_idx"))
    conn.execute(text(f"ALTER TABLE chunks ALTER COLUMN vector TYPE vector({int(embed_dim)})"))


def create_schema(embed_dim: int) -> bool:
    """Create tables, indexes and the pgvector column.

    Returns True when pgvector is usable. Raises RuntimeError on a non-Postgres URL.
    """
    from sqlalchemy import DDL

    engine = get_engine()
    dialect = engine.dialect.name
    if dialect != "postgresql":
        raise RuntimeError(
            f"MyRag requires PostgreSQL with pgvector; got '{dialect}'."
        )

    with engine.begin() as conn:
        vector_ok = pgvector_available(conn)

    if not vector_ok:
        raise RuntimeError(
            "The pgvector extension is not available. Install it (e.g. "
            "'CREATE EXTENSION vector' as a superuser) before starting MyRag."
        )

    metadata = _build_metadata(_column_type(dialect, embed_dim))
    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            DDL(
                "CREATE INDEX IF NOT EXISTS chunks_vector_hnsw_idx "
                "ON chunks USING hnsw (vector vector_cosine_ops)"
            )
        )
        # Lightweight migrations for columns added after the first release.
        conn.execute(
            DDL("ALTER TABLE query ADD COLUMN IF NOT EXISTS reasoning TEXT")
        )

        actual = vector_dim(conn)
        if actual is not None and actual != embed_dim:
            raise RuntimeError(
                f"The database column chunks.vector is vector({actual}) but EMBED_DIM="
                f"{embed_dim}. Changing the embedding dimension requires re-embedding "
                "every document: stop the server and run "
                "'python -m app.admin reembed --confirm' (add --embed-model MODEL to "
                "switch models at the same time)."
            )
    return vector_ok


def _build_metadata(vector_type):
    """Declarative metadata for every table (single source of truth for DDL)."""
    from sqlalchemy import (
        BigInteger,
        Boolean,
        Column,
        DateTime,
        Index,
        Integer,
        MetaData,
        Numeric,
        Table,
        Text,
        func,
    )
    from sqlalchemy import (
        text as sql_text,
    )
    from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
    from sqlalchemy.schema import Computed

    metadata = MetaData()

    Table(
        "machines",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("title", Text, nullable=False),
        Column("embed_model_id", Text, nullable=False),
        Column("embed_dim", Integer, nullable=False),
        Column("chunk_size", Integer, nullable=False),
        Column("chunk_overlap", Integer, nullable=False),
        Column("embed_tokens_per_char", Numeric(9, 6), nullable=True),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )

    Table(
        "documents",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("machine_id", BigInteger, nullable=False),
        Column("filename", Text, nullable=False),
        Column("content_type", Text, nullable=False),
        Column("byte_size", BigInteger, nullable=False),
        Column("stored_path", Text, nullable=True),
        Column("status", Text, nullable=False, server_default="uploaded"),
        Column("error", Text, nullable=True),
        Column("chunk_count", Integer, nullable=False, server_default="0"),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        Index("ix_documents_machine_id", "machine_id"),
    )

    Table(
        "chunks",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("machine_id", BigInteger, nullable=False),
        Column("document_id", BigInteger, nullable=False),
        Column("chunk_index", Integer, nullable=False),
        Column("text", Text, nullable=False),
        Column("metadata", JSONB, nullable=False, server_default=sql_text("'{}'::jsonb")),
        Column("embed_model_id", Text, nullable=True),
        Column("embed_dim", Integer, nullable=True),
        Column("vector", vector_type, nullable=True),
        Column("token_estimate", Integer, nullable=False, server_default="0"),
        Column(
            "search_tsv",
            TSVECTOR,
            Computed("to_tsvector('english', text)", persisted=True),
            nullable=True,
        ),
        Index("ix_chunks_machine_id", "machine_id"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_search_tsv", "search_tsv", postgresql_using="gin"),
    )

    Table(
        "workflows",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("name", Text, nullable=False, unique=True),
        Column("description", Text, nullable=False),
        Column("class_key", Text, nullable=False),
        Column("enabled", Boolean, nullable=False, server_default=sql_text("true")),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )

    Table(
        "query",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("machine_id", BigInteger, nullable=False),
        Column("workflow_id", BigInteger, nullable=False),
        Column("workflow_version", BigInteger, nullable=False),
        Column("llm_model_id", Text, nullable=False),
        Column("question", Text, nullable=False),
        Column("answer", Text, nullable=True),
        Column("reasoning", Text, nullable=True),
        Column("retrieval_ms", Integer, nullable=True),
        Column("total_ms", Integer, nullable=True),
        Column("prompt_tokens", Integer, nullable=False, server_default="0"),
        Column("completion_tokens", Integer, nullable=False, server_default="0"),
        Column("total_tokens", Integer, nullable=False, server_default="0"),
        Column("cost_usd", Numeric(18, 8), nullable=False, server_default="0"),
        Column("cost_source", Text, nullable=False, server_default="none"),
        Column("retrieved_chunk_ids", ARRAY(BigInteger), nullable=False, server_default="{}"),
        Column("error", Text, nullable=True),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        Index("ix_query_machine_created", "machine_id", "created_at"),
        Index("ix_query_workflow", "workflow_id"),
    )

    Table(
        "query_llm_call",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("query_id", BigInteger, nullable=False),
        Column("machine_id", BigInteger, nullable=False),
        Column("model_id", Text, nullable=False),
        Column("call_index", Integer, nullable=False),
        Column("purpose", Text, nullable=False),
        Column("prompt_tokens", Integer, nullable=False, server_default="0"),
        Column("completion_tokens", Integer, nullable=False, server_default="0"),
        Column("total_tokens", Integer, nullable=False, server_default="0"),
        Column("cost_usd", Numeric(18, 8), nullable=False, server_default="0"),
        Column("cost_source", Text, nullable=False, server_default="none"),
        Column("usage_raw", JSONB, nullable=True),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        Index("ix_query_llm_call_query", "query_id"),
    )

    Table(
        "query_event",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("query_id", BigInteger, nullable=False),
        Column("kind", Text, nullable=False),
        Column("payload", Text, nullable=False),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        Index("ix_query_event_query", "query_id"),
    )

    Table(
        "ingest_usage",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("machine_id", BigInteger, nullable=False),
        Column("document_id", BigInteger, nullable=False),
        Column("embed_model_id", Text, nullable=False),
        Column("char_count", BigInteger, nullable=False),
        Column("token_estimate", Integer, nullable=False),
        Column("cost_usd", Numeric(18, 8), nullable=False, server_default="0"),
        Column("cost_source", Text, nullable=False, server_default="estimated"),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
        Index("ix_ingest_usage_machine", "machine_id"),
    )

    return metadata