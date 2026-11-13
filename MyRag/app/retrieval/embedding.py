"""Dense vector retrieval over the machine's pgvector column."""

from __future__ import annotations

from sqlalchemy import text

from ..db import session_scope
from ..ingest.embeddings import embed_texts
from .base import RetrievedChunk, as_metadata_dict


class EmbeddingRetrievalFlow:
    name = "embedding"
    description = (
        "Dense cosine similarity over stored chunk vectors (pgvector HNSW index), "
        "filtered to the machine."
    )

    def retrieve(
        self, *, machine_id: int, question: str, top_k: int, embed_model_id: str
    ) -> list[RetrievedChunk]:
        vectors = embed_texts(embed_model_id, [question])
        if not vectors:
            return []
        embedding = vectors[0]

        vector_literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"

        sql = text(
            """
            SELECT c.id            AS chunk_id,
                   c.document_id   AS document_id,
                   c.text          AS text,
                   c.metadata      AS metadata,
                   1 - (c.vector <=> CAST(:embedding AS vector)) AS score
              FROM chunks c
             WHERE c.machine_id = :machine_id
               AND c.vector IS NOT NULL
             ORDER BY c.vector <=> CAST(:embedding AS vector)
             LIMIT :top_k
            """
        )

        with session_scope() as session:
            rows = session.execute(
                sql,
                {"embedding": vector_literal, "machine_id": machine_id, "top_k": top_k},
            ).mappings().all()

        results = []
        for position, row in enumerate(rows):
            results.append(
                RetrievedChunk(
                    chunk_id=int(row["chunk_id"]),
                    document_id=int(row["document_id"]),
                    text=row["text"] or "",
                    metadata=as_metadata_dict(row["metadata"]),
                    score=float(row["score"] or 0.0),
                    rank=position,
                )
            )
        return results