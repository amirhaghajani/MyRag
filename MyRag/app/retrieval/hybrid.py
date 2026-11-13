"""Hybrid retrieval: Postgres full text + dense vectors, fused with RRF."""

from __future__ import annotations

from sqlalchemy import text

from ..db import session_scope
from .base import RetrievedChunk, as_metadata_dict, reciprocal_rank_fusion
from .embedding import EmbeddingRetrievalFlow

_FULL_TEXT_SQL = text(
    """
    SELECT c.id            AS chunk_id,
           c.document_id   AS document_id,
           c.text          AS text,
           c.metadata      AS metadata,
           ts_rank(c.search_tsv, websearch_to_tsquery('english', :question)) AS score
      FROM chunks c
     WHERE c.machine_id = :machine_id
       AND c.search_tsv @@ websearch_to_tsquery('english', :question)
     ORDER BY score DESC
     LIMIT :top_k
    """
)


class HybridRetrievalFlow:
    name = "hybrid"
    description = (
        "Postgres full-text (tsvector + ts_rank) fused with dense pgvector results "
        "via reciprocal rank fusion."
    )

    def retrieve(
        self, *, machine_id: int, question: str, top_k: int, embed_model_id: str
    ) -> list[RetrievedChunk]:
        dense = EmbeddingRetrievalFlow().retrieve(
            machine_id=machine_id, question=question, top_k=top_k, embed_model_id=embed_model_id
        )
        sparse = self._sparse(machine_id=machine_id, question=question, top_k=top_k)
        return reciprocal_rank_fusion([dense, sparse], top_k=top_k)

    def _sparse(self, *, machine_id: int, question: str, top_k: int) -> list[RetrievedChunk]:
        cleaned = " ".join(question.split())
        if not cleaned:
            return []
        try:
            with session_scope() as session:
                rows = session.execute(
                    _FULL_TEXT_SQL,
                    {"machine_id": machine_id, "question": cleaned, "top_k": top_k},
                ).mappings().all()
        except Exception:  # noqa: BLE001 - sparse search is best-effort; dense still works
            # Search config may be missing on odd locales; dense results still stand.
            return []

        return [
            RetrievedChunk(
                chunk_id=int(row["chunk_id"]),
                document_id=int(row["document_id"]),
                text=row["text"] or "",
                metadata=as_metadata_dict(row["metadata"]),
                score=float(row["score"] or 0.0),
                rank=position,
            )
            for position, row in enumerate(rows)
        ]