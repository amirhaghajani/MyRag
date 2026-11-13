"""BM25 keyword retrieval over the machine's chunks."""

from __future__ import annotations

import logging

from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from ..db import session_scope
from .base import RetrievedChunk, as_metadata_dict

logger = logging.getLogger(__name__)

_cache: dict[tuple[int, int], BM25Retriever] = {}
_cache_order: list[tuple[int, int]] = []
_MAX_CACHE = 32


def _chunk_nodes(machine_id: int) -> list[TextNode]:
    from sqlalchemy import text

    sql = text(
        """
        SELECT c.id AS chunk_id, c.document_id, c.text, c.metadata
          FROM chunks c
         WHERE c.machine_id = :machine_id
         ORDER BY c.document_id, c.chunk_index
        """
    )
    with session_scope() as session:
        rows = session.execute(sql, {"machine_id": machine_id}).mappings().all()

    nodes = []
    for row in rows:
        metadata = as_metadata_dict(row["metadata"])
        metadata["_chunk_id"] = int(row["chunk_id"])
        metadata["_document_id"] = int(row["document_id"])
        nodes.append(TextNode(text=row["text"] or "", metadata=metadata))
    return nodes


def build_bm25(machine_id: int, top_k: int) -> BM25Retriever | None:
    """Build (or reuse) a BM25 index for a machine. Returns None when empty."""
    key = (machine_id, top_k)
    if key in _cache:
        return _cache[key]

    nodes = _chunk_nodes(machine_id)
    if not nodes:
        return None

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    _cache[key] = retriever
    _cache_order.append(key)
    while len(_cache_order) > _MAX_CACHE:
        stale = _cache_order.pop(0)
        _cache.pop(stale, None)
    return retriever


def invalidate(machine_id: int | None = None) -> None:
    """Drop cached indexes, for a single machine or all of them."""
    if machine_id is None:
        _cache.clear()
        _cache_order.clear()
        return
    for key in [k for k in _cache_order if k[0] == machine_id]:
        _cache.pop(key, None)
        _cache_order.remove(key)


class KeywordRetrievalFlow:
    name = "keyword"
    description = "BM25 lexical matching over chunk text, built from the machine's chunks."

    def retrieve(self, *, machine_id: int, question: str, top_k: int) -> list[RetrievedChunk]:
        retriever = build_bm25(machine_id, top_k)
        if retriever is None:
            return []

        nodes = retriever.retrieve(question)
        results = []
        for position, node in enumerate(nodes):
            metadata = dict(node.node.metadata or {})
            results.append(
                RetrievedChunk(
                    chunk_id=int(metadata.get("_chunk_id", 0)),
                    document_id=int(metadata.get("_document_id", 0)),
                    text=node.node.get_content() if hasattr(node, "node") else "",
                    metadata={k: v for k, v in metadata.items() if not k.startswith("_")},
                    score=float(node.score or 0.0),
                    rank=position,
                )
            )
        return results