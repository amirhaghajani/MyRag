"""Expose the configured models so the UI can offer a picker."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import get_settings
from ..schemas import ModelOption, ModelsOut

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelsOut)
def list_models() -> ModelsOut:
    settings = get_settings()

    llms = [
        ModelOption(
            id=price.model_id,
            input_per_1m=price.input_per_1m,
            output_per_1m=price.output_per_1m,
            context_window=price.context_window,
        )
        for price in settings.llm_prices.values()
    ]
    embeddings = [
        ModelOption(
            id=price.model_id,
            input_per_1m=price.input_per_1m,
            dim=price.dim,
        )
        for price in settings.embed_prices.values()
    ]

    return ModelsOut(
        llms=llms,
        embeddings=embeddings,
        llm_api_base=settings.api_base,
        embed_api_base=settings.embed_api_base_effective,
        default_llm_model=settings.default_llm_model,
        default_embed_model=settings.default_embed_model,
        embed_dim=settings.embed_dim,
    )