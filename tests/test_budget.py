from __future__ import annotations

import pytest

from optagent.budget import (
    BudgetLimits,
    _byte_pair_estimate,
    limits_from_price_table,
    precheck,
)


PRICE_TABLE = {
    "price_table_version": "test-1",
    "limits": {
        "max_input_tokens": 60000,
        "max_output_tokens": 2000,
        "max_retries": 2,
        "timeout_s": 45,
        "safety_margin": 0.20,
        "cap_usd": 0.05,
    },
    "models": {
        "claude-opus-4-7": {
            "input_usd_per_mtok": 15.0,
            "output_usd_per_mtok": 75.0,
            "tokenizer_version": "claude-2026-04",
        },
        "claude-haiku-4-5": {
            "input_usd_per_mtok": 0.80,
            "output_usd_per_mtok": 4.0,
            "tokenizer_version": "claude-2026-04",
        },
    },
}


def test_estimator_never_undercounts_empirically():
    # Conservative 2.5 chars/token; real Claude is closer to 3.5.
    text = "Hello world " * 50
    n = _byte_pair_estimate(text, model_version="claude-opus-4-7")
    assert n >= len(text) / 4  # generous lower bound


def test_limits_from_price_table_uses_yaml_values():
    lim = limits_from_price_table(PRICE_TABLE)
    assert lim.max_input_tokens == 60000
    assert lim.max_output_tokens == 2000
    assert lim.safety_margin == 0.20
    assert lim.cap_usd == 0.05


def test_precheck_proceeds_for_small_haiku_prompt():
    r = precheck(
        prompt_text="ticker=AAPL",
        model_version="claude-haiku-4-5",
        price_table=PRICE_TABLE,
        estimator=lambda t, model_version: 100,
    )
    assert r.proceed is True
    assert r.fallback_reason is None
    assert r.estimated_usd > 0
    assert r.tokenizer_version == "claude-2026-04"


def test_precheck_unknown_model_falls_back():
    r = precheck(
        prompt_text="x",
        model_version="gpt-5",
        price_table=PRICE_TABLE,
        estimator=lambda t, model_version: 10,
    )
    assert r.proceed is False
    assert r.fallback_reason == "unknown_model_pricing"


def test_precheck_input_token_cap_blocks():
    over_limit_table = dict(PRICE_TABLE)
    over_limit_table["limits"] = dict(PRICE_TABLE["limits"])
    over_limit_table["limits"]["max_input_tokens"] = 50
    r = precheck(
        prompt_text="x",
        model_version="claude-haiku-4-5",
        price_table=over_limit_table,
        estimator=lambda t, model_version: 1000,
    )
    assert r.proceed is False
    assert "input_tokens_exceeded" in (r.fallback_reason or "")


def test_precheck_cost_cap_blocks_expensive_opus():
    # opus pricing × 60k tokens > $0.05 cap, even before retries.
    r = precheck(
        prompt_text="x",
        model_version="claude-opus-4-7",
        price_table=PRICE_TABLE,
        estimator=lambda t, model_version: 60000,
    )
    assert r.proceed is False
    assert "budget_exceeded" in (r.fallback_reason or "")


def test_precheck_worst_case_includes_retries_and_safety_margin():
    cheap_table = dict(PRICE_TABLE)
    cheap_table["limits"] = {**PRICE_TABLE["limits"], "cap_usd": 100.0}
    r = precheck(
        prompt_text="x",
        model_version="claude-haiku-4-5",
        price_table=cheap_table,
        estimator=lambda t, model_version: 10_000,
    )
    # (10000+2000) tokens × 3 attempts × (input cost 10000/1e6 × 0.80 + output 2000/1e6 × 4.0) × 1.20
    # = (10000/1e6×0.80 + 2000/1e6×4.0) × 3 × 1.20 = (0.008 + 0.008) × 3 × 1.20 = 0.0576
    assert r.estimated_usd == pytest.approx(0.0576, abs=1e-4)
    assert r.proceed is True
