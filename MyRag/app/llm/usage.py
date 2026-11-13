"""Token accounting for LLM calls.

The handler collects usage emitted by llama_index's instrumentation dispatcher and
writes one ``query_llm_call`` row per call, tagged with the active correlation id.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
)

from .pricing import resolve_llm_cost

logger = logging.getLogger(__name__)


@dataclass
class PendingCall:
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cost_source: str
    usage_raw: dict = field(default_factory=dict)


_current_query_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "myrag_query_id", default=None
)


def set_current_query_id(query_id: int | None) -> contextvars.Token:
    return _current_query_id.set(query_id)


def reset_current_query_id(token: contextvars.Token) -> None:
    _current_query_id.reset(token)


def get_current_query_id() -> int | None:
    return _current_query_id.get()


def _as_dict(raw: object) -> dict:
    """Normalise a raw response that may be a dict or a Pydantic/SDK object."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:  # noqa: BLE001 - fall back to attribute access
            pass
    return {}


def _usage_from_raw(raw: object) -> dict:
    """Token usage from a dict or an SDK response object."""
    if isinstance(raw, dict):
        usage = raw.get("usage")
        if isinstance(usage, dict):
            return usage
        if usage is None:
            return {}
        usage_obj = usage
    else:
        usage_obj = getattr(raw, "usage", None)

    if usage_obj is None:
        return {}

    if isinstance(usage_obj, dict):
        return usage_obj

    dump = getattr(usage_obj, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:  # noqa: BLE001
            pass

    return {
        key: getattr(usage_obj, key, 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if getattr(usage_obj, key, None) is not None
    }


def _provider_cost_from_raw(raw: object) -> float | None:
    """Lift a provider-reported cost out of the raw response when present."""
    data = _as_dict(raw)
    if not data:
        return None

    for key in ("cost", "total_cost", "usage_cost", "_myrag_cost"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    usage = data.get("usage")
    if isinstance(usage, dict):
        for key in ("cost", "total_cost"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _prompt_completion_from_usage(usage: dict) -> tuple[int, int]:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return prompt, completion, total


class UsageEventHandler(BaseEventHandler):
    """Turns LLM end events into PendingCall records for the enclosing query."""

    calls: list[PendingCall] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "MyRagUsageEventHandler"

    def handle(self, event: BaseEvent, **kwargs) -> None:  # type: ignore[override]
        if not isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)):
            return
        query_id = _current_query_id.get()
        if query_id is None:
            return

        response = event.response
        raw = getattr(response, "raw", None) if response is not None else None
        usage = _usage_from_raw(raw)
        prompt_tokens, completion_tokens, total_tokens = _prompt_completion_from_usage(usage)

        model_id = self._model_id(event, raw)
        provider_cost = _provider_cost_from_raw(raw)
        cost_result = resolve_llm_cost(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider_cost=provider_cost,
        )

        self.calls.append(
            PendingCall(
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_result.usd,
                cost_source=cost_result.source,
                usage_raw=usage or _as_dict(raw),
            )
        )

    def _model_id(self, event: BaseEvent, raw: object) -> str:
        data = _as_dict(raw)
        if data.get("model"):
            return str(data["model"])
        model_dict = getattr(event, "model_dict", None)
        if isinstance(model_dict, dict) and model_dict.get("model_name"):
            return str(model_dict["model_name"])
        return "unknown"

    def drain(self) -> list[PendingCall]:
        calls, self.calls = self.calls, []
        return calls


_handler: UsageEventHandler | None = None


def install_usage_handler() -> UsageEventHandler:
    """Register the handler once with the global dispatcher."""
    global _handler
    if _handler is None:
        _handler = UsageEventHandler()
        get_dispatcher().add_event_handler(_handler)
    return _handler


def get_usage_handler() -> UsageEventHandler:
    return install_usage_handler()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()