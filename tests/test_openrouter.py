"""OpenRouter provider wiring (no network)."""

from __future__ import annotations

import pytest


_KEYS = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _clear(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def test_chat_detects_openrouter_first(monkeypatch):
    from optagent.web import chat

    _clear(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    # OpenRouter wins even when a single-vendor key is also present.
    assert chat._detect_provider() == "openrouter"


def test_chat_default_openrouter_model_env_override(monkeypatch):
    from optagent.web import chat

    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert chat._openrouter_model() == "anthropic/claude-sonnet-4.6"
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5.5")
    assert chat._openrouter_model() == "openai/gpt-5.5"


def test_chat_openrouter_in_default_models():
    from optagent.web import chat

    assert "openrouter" in chat._DEFAULT_MODELS


def test_make_client_from_env_picks_openrouter(monkeypatch):
    from optagent import llm

    _clear(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    captured = {}

    def _fake_factory(model="anthropic/claude-sonnet-4.6"):
        captured["model"] = model
        return object()

    monkeypatch.setattr(llm, "make_openrouter_client", _fake_factory)
    client, provider, model = llm.make_client_from_env()
    assert provider == "openrouter"
    assert model == "anthropic/claude-sonnet-4.6"
    assert captured["model"] == "anthropic/claude-sonnet-4.6"


def test_make_client_from_env_openrouter_model_override(monkeypatch):
    from optagent import llm

    _clear(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5.5")
    monkeypatch.setattr(llm, "make_openrouter_client", lambda model="x": object())
    _client, provider, model = llm.make_client_from_env()
    assert provider == "openrouter"
    assert model == "openai/gpt-5.5"


def test_make_openrouter_client_requires_key(monkeypatch):
    from optagent import llm

    _clear(monkeypatch)
    with pytest.raises(RuntimeError) as ei:
        llm.make_openrouter_client(api_key=None)
    assert "OPENROUTER_API_KEY" in str(ei.value)
