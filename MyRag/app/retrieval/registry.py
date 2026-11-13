"""Registry mapping a workflow name to its retrieval flow class."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .base import RetrievalFlow, RetrievedChunk
from .embedding import EmbeddingRetrievalFlow
from .hybrid import HybridRetrievalFlow
from .keyword import KeywordRetrievalFlow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowSpec:
    class_key: str
    factory: type
    needs_embed_model: bool


# Adding a v2 of a flow: define a new class, add it here, and seed a workflows row.
WORKFLOWS: dict[str, WorkflowSpec] = {
    "embedding": WorkflowSpec("embedding", EmbeddingRetrievalFlow, True),
    "keyword": WorkflowSpec("keyword", KeywordRetrievalFlow, False),
    "hybrid": WorkflowSpec("hybrid", HybridRetrievalFlow, True),
}


class UnknownWorkflowError(KeyError):
    pass


def build_flow(class_key: str) -> RetrievalFlow:
    spec = WORKFLOWS.get(class_key)
    if spec is None:
        raise UnknownWorkflowError(
            f"Unknown workflow class key '{class_key}'. Known: {', '.join(sorted(WORKFLOWS))}"
        )
    return spec.factory()  # type: ignore[return-value]


def spec_for(class_key: str) -> WorkflowSpec:
    spec = WORKFLOWS.get(class_key)
    if spec is None:
        raise UnknownWorkflowError(class_key)
    return spec


def retrieve(
    *,
    class_key: str,
    machine_id: int,
    question: str,
    top_k: int,
    embed_model_id: str,
) -> list[RetrievedChunk]:
    """Dispatch to the right flow, passing the embed model only when it needs one."""
    spec = spec_for(class_key)
    flow = build_flow(class_key)

    if spec.needs_embed_model:
        return flow.retrieve(  # type: ignore[call-arg]
            machine_id=machine_id, question=question, top_k=top_k, embed_model_id=embed_model_id
        )
    return flow.retrieve(machine_id=machine_id, question=question, top_k=top_k)  # type: ignore[call-arg]


__all__ = [
    "WORKFLOWS",
    "RetrievedChunk",
    "UnknownWorkflowError",
    "WorkflowSpec",
    "build_flow",
    "retrieve",
    "spec_for",
]