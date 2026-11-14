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


def vector_column(dim: int) -> str:
    """Name of the per-dimension vector column."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive.")
    return f"vector_{dim}"


def existing_vector_dims(conn) -> dict[int, str]:
    """Map dimension -> column name for the vector_<dim> columns that exist."""
    rows = conn.execute(
        text(
            "SELECT a.attname, a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = 'chunks' AND a.attnum > 0 "
            "AND a.attname LIKE 'vector\\_%' "
            "AND format_type(a.atttypid, a.atttypmod) LIKE 'vector(%'"
        )
    ).mappings().all()
    return {int(row["atttypmod"]): str(row["attname"]) for row in rows}


def ensure_vector_column(conn, dim: int) -> str:
    """Add the vector_<dim> column and its HNSW index when missing."""
    column = vector_column(dim)
    conn.execute(
        text(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {column} vector({int(dim)})")
    )
    conn.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS chunks_{column}_hnsw_idx "
            f"ON chunks USING hnsw ({column} vector_cosine_ops)"
        )
    )
    return column


def configured_dims() -> list[int]:
    from .config import get_settings

    return sorted({int(price.dim) for price in get_settings().embed_prices.values()})


def migrate_legacy_vectors(conn) -> int:
    """Copy rows from the pre-multi-dim chunks.vector column into vector_<dim>.

    Returns the number of rows migrated. The legacy column is left in place so the
    migration can be rolled back by keeping the old code path.
    """
    has_legacy = conn.execute(
        text(
            "SELECT 1 FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = 'chunks' AND a.attname = 'vector' AND a.attnum > 0"
        )
    ).first()
    if has_legacy is None:
        return 0

    dims = conn.execute(
        text(
            "SELECT DISTINCT embed_dim FROM chunks "
            "WHERE vector IS NOT NULL AND embed_dim IS NOT NULL"
        )
    ).scalars().all()

    moved = 0
    for dim in dims:
        column = ensure_vector_column(conn, int(dim))
        result = conn.execute(
            text(
                f"UPDATE chunks SET {column} = vector "
                f"WHERE embed_dim = :dim AND {column} IS NULL AND vector IS NOT NULL"
            ),
            {"dim": int(dim)},
        )
        moved += result.rowcount or 0
    return moved


def create_schema() -> bool:
    """Create tables, per-dimension vector columns and indexes.

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

    metadata = _build_metadata()
    metadata.create_all(engine)

    with engine.begin() as conn:
        # Lightweight migrations for columns added after the first release.
        conn.execute(
            DDL("ALTER TABLE query ADD COLUMN IF NOT EXISTS reasoning TEXT")
        )

        dims = set(configured_dims())
        try:
            rows = conn.execute(text("SELECT embed_dim FROM machines")).all()
            dims.update(int(row[0]) for row in rows if row[0] is not None)
        except Exception:  # noqa: BLE001 - machines table may not exist yet
            pass

        for dim in sorted(dims):
            ensure_vector_column(conn, dim)

        migrate_legacy_vectors(conn)
    return vector_ok


def _build_metadata():
    """Declarative metadata for every table (single source of truth for DDL).

    Vector columns are managed separately by ensure_vector_column(): one
    vector_<dim> column per embedding dimension, so machines can use different models.
    """
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