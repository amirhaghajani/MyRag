"""Shared retrieval types and the flow protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def as_metadata_dict(value: object) -> dict:
    """Normalise a metadata column that may arrive as dict or JSON text.

    Postgres returns jsonb as a dict; SQLite (used in tests) returns text.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0


@runtime_checkable
class RetrievalFlow(Protocol):
    name: str
    description: str

    def retrieve(
        self, *, machine_id: int, question: str, top_k: int
    ) -> list[RetrievedChunk]:
        """Return the top chunks for a question, best first."""
        ...


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int = 4,
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one using reciprocal rank fusion."""
    scores: dict[int, float] = {}
    best: dict[int, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for position, chunk in enumerate(ranked):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + position + 1)
            if chunk.chunk_id not in best:
                best[chunk.chunk_id] = chunk

    fused = []
    for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        chunk = best[chunk_id]
        fused.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=score,
                rank=len(fused),
            )
        )
        if len(fused) >= top_k:
            break
    return fused