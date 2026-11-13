"""Cached LLM construction from the .env model list."""

from __future__ import annotations

import logging
from functools import lru_cache

from llama_index.core.llms import LLM
from llama_index.llms.openai_like import OpenAILike

from ..config import get_settings

logger = logging.getLogger(__name__)

_COST_KEYS = ("cost", "total_cost", "usage_cost")


class CostAwareOpenAILike(OpenAILike):
    """OpenAILike that lifts a provider-reported cost into ``_myrag_cost``.

    Some OpenAI-compatible gateways (OpenRouter) return the exact charged amount.
    The standard client drops it, so it is copied onto the raw response dict where
    the usage handler can read it without re-parsing.
    """

    def _extract_cost(self, raw: object) -> float | None:
        if not isinstance(raw, dict):
            return None
        for key in _COST_KEYS:
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        usage = raw.get("usage")
        if isinstance(usage, dict):
            for key in ("cost", "total_cost"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    def chat(self, messages, **kwargs):  # type: ignore[override]
        response = super().chat(messages, **kwargs)
        cost = self._extract_cost(getattr(response, "raw", None))
        if cost is not None and isinstance(response.raw, dict):
            response.raw["_myrag_cost"] = cost
        return response

    async def achat(self, messages, **kwargs):  # type: ignore[override]
        response = await super().achat(messages, **kwargs)
        cost = self._extract_cost(getattr(response, "raw", None))
        if cost is not None and isinstance(response.raw, dict):
            response.raw["_myrag_cost"] = cost
        return response


@lru_cache(maxsize=64)
def build_llm(model_id: str) -> LLM:
    """Build (and cache) an LLM for a model id declared in LLM_MODELS."""
    settings = get_settings()
    price = settings.llm_prices.get(model_id)
    if price is None:
        known = ", ".join(sorted(settings.llm_prices))
        raise ValueError(f"Unknown LLM model '{model_id}'. Configured: {known}")

    return CostAwareOpenAILike(
        model=model_id,
        api_base=settings.api_base,
        api_key=settings.api_key or "not-needed",
        context_window=price.context_window,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0.1,
        timeout=120.0,
        max_retries=2,
    )