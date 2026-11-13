"""Price lookups and cost resolution.

Prices come from .env per model. When the provider reports an exact cost on the
response, that value wins and is marked ``provider``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import EmbedPrice, LlmPrice, get_settings

logger = logging.getLogger(__name__)

_SAMPLE_CHARS = 2048
_ENCODER = None


@dataclass(frozen=True)
class CostResult:
    usd: float
    source: str


def price_for_llm(model_id: str) -> LlmPrice | None:
    return get_settings().llm_prices.get(model_id)


def price_for_embed(model_id: str) -> EmbedPrice | None:
    return get_settings().embed_prices.get(model_id)


def resolve_llm_cost(
    *,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    provider_cost: float | None = None,
) -> CostResult:
    """Provider-reported cost wins; otherwise .env price x measured tokens."""
    if provider_cost is not None and provider_cost >= 0:
        return CostResult(provider_cost, "provider")

    price = price_for_llm(model_id)
    if price is None:
        return CostResult(0.0, "none")

    usd = (
        prompt_tokens * price.input_per_1m
        + completion_tokens * price.output_per_1m
    ) / 1_000_000
    return CostResult(usd, "env")


def _get_encoder():
    """Return a tiktoken encoder, or None when tiktoken is unavailable."""
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    try:
        import tiktoken
    except ImportError:
        logger.info("tiktoken not installed; embedding tokens fall back to chars/4.")
        return None
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - offline environments cannot download the encoding
        logger.info("tiktoken encoder unavailable; using chars/4 fallback.")
        return None
    return _ENCODER


def tokens_per_char() -> float:
    """Calibrated tokens-per-character ratio for the fallback tokenizer."""
    encoder = _get_encoder()
    sample = "the quick brown fox jumps over the lazy dog " * 32
    if encoder is not None:
        return len(encoder.encode(sample)) / len(sample)
    return 0.25


def estimate_embed_tokens(char_count: int, ratio: float | None = None) -> int:
    """Estimate embedding tokens from a character count.

    Embedding responses expose no token counts, so this is always an estimate.
    A previously calibrated ``ratio`` (stored per machine) is preferred.
    """
    if char_count <= 0:
        return 0

    if ratio is None or ratio <= 0:
        encoder = _get_encoder()
        if encoder is not None:
            return len(encoder.encode("x" * char_count))
        return max(1, char_count // 4)

    return max(1, round(char_count * ratio))


def resolve_embed_cost(
    *,
    model_id: str,
    char_count: int,
    ratio: float | None = None,
) -> tuple[CostResult, int]:
    """Return (cost, token_estimate). Always ``estimated`` or ``none``."""
    token_estimate = estimate_embed_tokens(char_count, ratio)
    price = price_for_embed(model_id)
    if price is None:
        return CostResult(0.0, "none"), token_estimate

    usd = token_estimate * price.input_per_1m / 1_000_000
    return CostResult(usd, "estimated"), token_estimate