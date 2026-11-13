"""Embedding model construction and batch embedding."""

from __future__ import annotations

import logging

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from ..config import get_settings

logger = logging.getLogger(__name__)

_cache: dict[str, BaseEmbedding] = {}


def build_embed_model(model_id: str) -> BaseEmbedding:
    """Build (and cache) the embedding client for an EMBED_MODELS entry.

    The embedding endpoint is configured independently from the LLM endpoint, so a
    separate API base and key are used here.
    """
    if model_id in _cache:
        return _cache[model_id]

    settings = get_settings()
    price = settings.embed_prices.get(model_id)
    if price is None:
        known = ", ".join(sorted(settings.embed_prices))
        raise ValueError(f"Unknown embed model '{model_id}'. Configured: {known}")

    if not settings.embed_api_base_effective:
        raise ValueError("EMBED_API_BASE (or API_BASE fallback) is required.")

    _cache[model_id] = OpenAILikeEmbedding(
        model_name=model_id,
        api_base=settings.embed_api_base_effective,
        api_key=settings.embed_api_key_effective or "not-needed",
        dimensions=price.dim,
        embed_batch_size=100,
        timeout=120.0,
        max_retries=3,
    )
    return _cache[model_id]


def embed_texts(model_id: str, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per text."""
    if not texts:
        return []
    model = build_embed_model(model_id)
    vectors: list[list[float]] = []
    batch_size = 96
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(model.get_text_embedding_batch(batch))
    return vectors


def check_embed_model(model_id: str) -> int:
    """Probe an embedding model and return the observed dimension."""
    vectors = embed_texts(model_id, ["dimension probe"])
    if not vectors or not vectors[0]:
        raise ValueError(f"Embedding model '{model_id}' returned no vector.")
    return len(vectors[0])