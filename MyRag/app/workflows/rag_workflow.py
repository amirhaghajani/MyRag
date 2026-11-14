"""RAG query flow built on llama_index workflows.

Steps: retrieve (registered flow) -> synthesize (LLM answer over the chunks) -> stop.

Note: this module deliberately does not use ``from __future__ import annotations`` —
the workflow step decorator resolves annotations at class-creation time and string
annotations break event detection.
"""

import logging
import time
from dataclasses import field

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from ..llm.factory import build_llm
from ..llm.trace import record_prompt, record_retrieval
from ..retrieval import registry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a technician assistant for flour factory machinery. "
    "Answer using only the provided machine documentation. "
    "If the answer is not in the documentation, say so plainly instead of guessing. "
    "Mention which document and section you used when it is relevant."
)


def _reasoning_of(response) -> str:
    """Pull the provider's reasoning/thinking text out of a chat response.

    Ollama returns it as ``reasoning``, some OpenAI-compatible gateways as
    ``reasoning_content``. Both arrive as extra fields on the raw message.
    """
    raw = getattr(response, "raw", None)
    if raw is None:
        return ""

    choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(
        choices[0], "message", None
    )
    if message is None:
        return ""

    extras = {}
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        extras.update(model_extra)

    for key in ("reasoning_content", "reasoning", "thinking"):
        value = extras.get(key)
        if value is None and isinstance(message, dict):
            value = message.get(key)
        if value is None:
            value = getattr(message, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_messages(question: str, chunks: list) -> list:
    """System prompt plus the retrieved chunks inlined as context."""
    context_parts = []
    for chunk in chunks:
        source = chunk.metadata.get("filename", "document")
        context_parts.append(f"[{source} #{chunk.chunk_id}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_parts) if context_parts else "(no chunks retrieved)"
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Documentation:\n\n{context}\n\nQuestion: {question}",
        ),
    ]


class RetrievedEvent(Event):
    question: str
    chunks: list = field(default_factory=list)
    retrieval_ms: int = 0
    llm_model_id: str = ""


class RagWorkflow(Workflow):
    """Retrieve chunks for a question, then answer with the selected LLM.

    Query settings travel on the start event rather than in workflow state so a run
    is fully described by its input.
    """

    @step
    async def retrieve(self, ev: StartEvent) -> RetrievedEvent:
        question = ev.get("question", "")
        started = time.perf_counter()
        chunks = registry.retrieve(
            class_key=ev.get("workflow_class_key", ""),
            machine_id=int(ev.get("machine_id", 0)),
            question=question,
            top_k=int(ev.get("top_k", 4)),
            embed_model_id=ev.get("embed_model_id", ""),
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.debug("retrieval returned %d chunks in %dms", len(chunks), elapsed_ms)
        record_retrieval(chunks)
        return RetrievedEvent(
            question=question,
            chunks=chunks,
            retrieval_ms=elapsed_ms,
            llm_model_id=ev.get("llm_model_id", ""),
        )

    @step
    async def synthesize(self, ev: RetrievedEvent) -> StopEvent:
        llm = build_llm(ev.llm_model_id)
        messages = build_messages(ev.question, ev.chunks)
        record_prompt(messages)
        response = await llm.achat(messages)
        return StopEvent(
            result={
                "answer": response.message.content or "",
                "reasoning": _reasoning_of(response),
                "chunks": ev.chunks,
                "retrieval_ms": ev.retrieval_ms,
            }
        )