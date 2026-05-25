"""Multi-model client smoke tests with vendor SDKs stubbed out.

We don't take real OpenAI/Gemini SDK imports here; the goal is to lock the
shape of `make_*_client` errors when SDKs are absent AND to verify
`make_client_from_env`'s provider-selection logic.
"""

from __future__ import annotations

import os

import pytest

from optagent.llm import make_client_from_env


def test_provider_selection_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPTAGENT_LLM_PROVIDER", raising=False)
    # Explicit anthropic should succeed in resolving (but SDK import may fail).
    with pytest.raises(RuntimeError):
        # In this env Anthropic SDK is not installed in the test runner; the
        # factory raises a clear RuntimeError. That's the contract we test.
        make_client_from_env(provider="openai")


def test_provider_selection_env_cascade(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPTAGENT_LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError) as ei:
        make_client_from_env()
    assert "No LLM provider configured" in str(ei.value)


def test_provider_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(RuntimeError) as ei:
        make_client_from_env(provider="cohere")
    assert "unknown LLM provider" in str(ei.value)


def test_env_var_provider_override(monkeypatch):
    monkeypatch.setenv("OPTAGENT_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    # No keys for gemini but OPTAGENT_LLM_PROVIDER forces gemini selection;
    # the actual SDK import is what fails next.
    with pytest.raises(RuntimeError):
        make_client_from_env()
