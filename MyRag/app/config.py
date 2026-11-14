"""Application settings loaded from the environment (.env is read automatically)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsError(RuntimeError):
    """Raised when the configuration cannot be used as written."""


@dataclass(frozen=True)
class LlmPrice:
    model_id: str
    input_per_1m: float
    output_per_1m: float
    context_window: int


@dataclass(frozen=True)
class EmbedPrice:
    model_id: str
    dim: int
    input_per_1m: float


def _parse_pairs(raw: str) -> dict[str, str]:
    """Split a 'a=1;b=2' style env value into ordered pairs."""
    pairs: dict[str, str] = {}
    for item in (raw or "").split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


def parse_llm_models(raw: str) -> list[LlmPrice]:
    """Parse LLM_MODELS: 'model_id=in,out,ctx; other=...' (prices per 1M tokens)."""
    models: list[LlmPrice] = []
    for model_id, spec in _parse_pairs(raw).items():
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) < 2:
            raise SettingsError(
                f"LLM_MODELS entry '{model_id}' must be 'in,out[,ctx]', got '{spec}'."
            )
        try:
            input_price, output_price = float(parts[0]), float(parts[1])
            context_window = int(parts[2]) if len(parts) > 2 else 128000
        except ValueError as exc:
            raise SettingsError(f"LLM_MODELS entry '{model_id}' has bad numbers: {exc}") from exc
        models.append(LlmPrice(model_id, input_price, output_price, context_window))
    return models


def parse_embed_models(raw: str) -> list[EmbedPrice]:
    """Parse EMBED_MODELS: 'model_id=dim,in; other=...'."""
    models: list[EmbedPrice] = []
    for model_id, spec in _parse_pairs(raw).items():
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) < 2:
            raise SettingsError(
                f"EMBED_MODELS entry '{model_id}' must be 'dim,in', got '{spec}'."
            )
        try:
            dim, input_price = int(parts[0]), float(parts[1])
        except ValueError as exc:
            raise SettingsError(f"EMBED_MODELS entry '{model_id}' has bad numbers: {exc}") from exc
        models.append(EmbedPrice(model_id, dim, input_price))
    return models


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/myragapp"

    api_base: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    llm_models: str = "openai/gpt-4o-mini=0.15,0.60,128000"
    default_llm_model: str = "openai/gpt-4o-mini"

    embed_api_base: str = ""
    embed_api_key: str = ""
    embed_models: str = "openai/text-embedding-3-small=1536,0.02"
    default_embed_model: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 4
    max_upload_mb: int = 25
    max_chunks_per_doc: int = 2000

    host: str = "127.0.0.1"
    port: int = 8000

    # Simple HTTP Basic auth. Auth is active only when both are set.
    auth_username: str = ""
    auth_password: str = ""

    @field_validator("embed_dim")
    @classmethod
    def _positive_dim(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("EMBED_DIM must be a positive integer.")
        return value

    @property
    def llm_prices(self) -> dict[str, LlmPrice]:
        return {m.model_id: m for m in parse_llm_models(self.llm_models)}

    @property
    def embed_prices(self) -> dict[str, EmbedPrice]:
        return {m.model_id: m for m in parse_embed_models(self.embed_models)}

    def validate_embed_dims(self) -> None:
        """Every configured embed model must match EMBED_DIM (one vector column)."""
        mismatches = [
            f"{m.model_id} (dim={m.dim})"
            for m in parse_embed_models(self.embed_models)
            if m.dim != self.embed_dim
        ]
        if mismatches:
            raise SettingsError(
                f"EMBED_DIM={self.embed_dim} does not match configured embed model(s): "
                f"{', '.join(mismatches)}. Every EMBED_MODELS entry must use "
                f"dim={self.embed_dim} so the single chunks.vector column stays valid."
            )

    def validate_models(self) -> None:
        if (self.auth_username == "") != (self.auth_password == ""):
            raise SettingsError(
                "AUTH_USERNAME and AUTH_PASSWORD must be set together (both or neither)."
            )
        if not self.api_base:
            raise SettingsError("API_BASE is required for the LLM endpoint.")
        if not self.llm_prices:
            raise SettingsError("LLM_MODELS must define at least one model.")
        if not self.embed_prices:
            raise SettingsError("EMBED_MODELS must define at least one model.")
        if self.default_llm_model not in self.llm_prices:
            raise SettingsError(
                f"DEFAULT_LLM_MODEL='{self.default_llm_model}' is not in LLM_MODELS."
            )
        if self.default_embed_model not in self.embed_prices:
            raise SettingsError(
                f"DEFAULT_EMBED_MODEL='{self.default_embed_model}' is not in EMBED_MODELS."
            )
        self.validate_embed_dims()

    @property
    def auth_enabled(self) -> bool:
        """Auth is active only when both credentials are configured."""
        return bool(self.auth_username and self.auth_password)

    @property
    def embed_api_base_effective(self) -> str:
        """Embedding endpoint falls back to the LLM endpoint when not configured."""
        return self.embed_api_base or self.api_base

    @property
    def embed_api_key_effective(self) -> str:
        return self.embed_api_key or self.api_key

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()