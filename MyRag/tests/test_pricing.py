import pytest

from app.config import Settings, SettingsError, parse_embed_models, parse_llm_models
from app.llm import pricing


def make_settings(**overrides) -> Settings:
    base = {
        "llm_models": "openai/gpt-4o-mini=0.15,0.60,128000; openai/gpt-4o=2.50,10.00,128000",
        "embed_models": "openai/text-embedding-3-small=1536,0.02",
        "embed_dim": 1536,
        "default_llm_model": "openai/gpt-4o-mini",
        "default_embed_model": "openai/text-embedding-3-small",
        "api_base": "https://openrouter.ai/api/v1",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    """Point pricing at a known configuration for every test in this module."""
    settings = make_settings()
    monkeypatch.setattr(pricing, "get_settings", lambda: settings)
    yield


def test_parse_llm_models_reads_price_and_context_window():
    models = parse_llm_models("a/b=0.15,0.60,128000; c/d=1.0,2.0")
    assert [m.model_id for m in models] == ["a/b", "c/d"]
    assert models[0].input_per_1m == 0.15
    assert models[0].output_per_1m == 0.60
    assert models[0].context_window == 128000
    # Context window defaults when omitted.
    assert models[1].context_window == 128000


def test_parse_embed_models_reads_dim_and_price():
    models = parse_embed_models("openai/text-embedding-3-small=1536,0.02")
    assert models[0].dim == 1536
    assert models[0].input_per_1m == 0.02


def test_bad_llm_spec_raises():
    with pytest.raises(SettingsError):
        parse_llm_models("a/b=onlyone")


def test_bad_embed_spec_raises():
    with pytest.raises(SettingsError):
        parse_embed_models("a/b=notanumber,1")


def test_env_price_is_used_when_no_provider_cost():
    result = pricing.resolve_llm_cost(
        model_id="openai/gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        provider_cost=None,
    )
    assert result.source == "env"
    assert result.usd == pytest.approx(0.15)


def test_env_price_splits_prompt_and_completion():
    result = pricing.resolve_llm_cost(
        model_id="openai/gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        provider_cost=None,
    )
    assert result.usd == pytest.approx(0.75)


def test_provider_cost_wins_over_env():
    result = pricing.resolve_llm_cost(
        model_id="openai/gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        provider_cost=0.42,
    )
    assert result.source == "provider"
    assert result.usd == pytest.approx(0.42)


def test_unknown_model_yields_none_source():
    result = pricing.resolve_llm_cost(
        model_id="nope/model",
        prompt_tokens=10,
        completion_tokens=10,
    )
    assert result.source == "none"
    assert result.usd == 0.0


def test_embed_cost_is_marked_estimated():
    cost, tokens = pricing.resolve_embed_cost(
        model_id="openai/text-embedding-3-small", char_count=4000
    )
    assert cost.source == "estimated"
    assert tokens > 0
    assert cost.usd > 0


def test_embed_tokens_monotonic_in_length():
    short = pricing.estimate_embed_tokens(100)
    long = pricing.estimate_embed_tokens(5000)
    assert long > short > 0


def test_embed_zero_chars_costs_nothing():
    cost, tokens = pricing.resolve_embed_cost(
        model_id="openai/text-embedding-3-small", char_count=0
    )
    assert tokens == 0
    assert cost.usd == 0.0


def test_tokens_per_char_fallback_is_bounded(monkeypatch):
    monkeypatch.setattr(pricing, "_ENCODER", None)
    monkeypatch.setattr(pricing, "_get_encoder", lambda: None)
    assert pricing.estimate_embed_tokens(400) == 100  # chars / 4


def test_embed_models_can_use_different_dims():
    settings = make_settings(embed_models="openai/a=1536,0.02; openai/b=3072,0.13")
    assert {m.dim for m in settings.embed_prices.values()} == {1536, 3072}


def test_validate_models_rejects_unknown_default_llm():
    settings = make_settings(default_llm_model="missing/model")
    with pytest.raises(SettingsError):
        settings.validate_models()


def test_embed_endpoint_falls_back_to_llm_endpoint():
    settings = make_settings(embed_api_base="", api_base="https://llm.example/v1")
    assert settings.embed_api_base_effective == "https://llm.example/v1"


def test_embed_endpoint_independent_of_llm_endpoint():
    settings = make_settings(
        embed_api_base="https://embed.example/v1",
        embed_api_key="embed-key",
        api_key="llm-key",
    )
    assert settings.embed_api_base_effective == "https://embed.example/v1"
    assert settings.embed_api_key_effective == "embed-key"
    assert settings.embed_api_key_effective != settings.api_key