"""Verify ml_signal block uses delimiter wrapping (Codex R4 finding)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from optagent.llm import SYSTEM_PROMPT, build_user_prompt
from optagent.schemas import (
    Confidence,
    Envelope,
    MarketSession,
    OptionContract,
    OptionRight,
)


UTC_NOW = datetime.now(timezone.utc)


def _envelope() -> Envelope:
    return Envelope(
        value={"last": 190.0},
        as_of=UTC_NOW,
        source="yfinance",
        delay_assumption="delayed_15min",
        market_session=MarketSession.rth,
        confidence=Confidence.ok,
        provider_profile_id="yfinance_research",
    )


def _contract() -> OptionContract:
    return OptionContract(
        occ_symbol="AAPL_C200",
        underlying="AAPL",
        expiration=UTC_NOW + timedelta(days=21),
        strike=200.0,
        right=OptionRight.call,
        mid=2.50,
        bid=2.45,
        ask=2.55,
        spread_pct=0.04,
        oi=12000,
        volume=1500,
        delta=0.42,
        theta=-0.05,
        vega=0.18,
        iv=0.28,
        breakeven=202.50,
        max_loss=250.0,
        liquidity_score=0.85,
        data_quality_score=0.95,
    )


def test_ml_signal_wrapped_in_delimiter_block():
    ml_signal = {
        "prob_up": 0.71,
        "class_label": "up",
        "credibility": "low",
        "oos_accuracy": 0.52,
        "wilson_ci_lower": 0.45,
        "wilson_ci_upper": 0.59,
        "class_baseline_accuracy": 0.53,
        "n_oos_samples": 95,
        "n_oos_folds": 5,
        "model_version": "ml-direction-v0",
        "feature_snapshot": {"rsi_14": 60.0},
    }
    prompt = build_user_prompt(
        ticker="AAPL",
        spot=190.0,
        candidates=[_contract()],
        envelopes=[_envelope()],
        ml_signal=ml_signal,
    )
    assert '<ml_signal id="ml-direction-v0">' in prompt
    assert "</ml_signal>" in prompt
    # The sensitive numeric fields land inside the delimiter block.
    open_idx = prompt.index('<ml_signal id="ml-direction-v0">')
    close_idx = prompt.index("</ml_signal>", open_idx)
    block = prompt[open_idx:close_idx]
    assert "prob_up: 0.71" in block
    assert "wilson_ci_95" in block
    assert "class_baseline_accuracy" in block


def test_system_prompt_declares_ml_signal_as_data():
    """Codex R4: SYSTEM_PROMPT must explicitly tell the LLM that ml_signal
    is auxiliary data, not instructions."""

    text = SYSTEM_PROMPT
    assert "<ml_signal>" in text
    assert "AUXILIARY" in text or "auxiliary" in text
    assert "NOT defer" in text or "sole or dominant" in text


def test_prompt_without_ml_signal_omits_block():
    prompt = build_user_prompt(
        ticker="AAPL",
        spot=190.0,
        candidates=[_contract()],
        envelopes=[_envelope()],
    )
    assert "<ml_signal" not in prompt


def test_ml_signal_id_citation_is_caught_by_validator():
    """Codex R5: explicit regression test for the LLM trying to cite the
    `ml-direction-v0` signal id as if it were a real tool_call_id. The
    validator's check (c) must catch this and downgrade to SKIP.
    """

    from optagent.profiles import ensure_default_profiles
    from optagent.registry import ProviderRegistry
    from optagent.render import render_template
    from optagent.schemas import (
        Citation,
        OptionContract,
        OptionRight,
        RunConfig,
        SkipReason,
        Verdict,
        VerdictAction,
    )
    from optagent.validator import validate

    registry = ProviderRegistry()
    ensure_default_profiles(registry)
    registry.bind(RunConfig(ticker="AAPL"))

    real_env = _envelope()
    contract = _contract()
    v = Verdict(
        disclaimer="RESEARCH ONLY — NOT FINANCIAL ADVICE.",
        action=VerdictAction.long_call,
        contract=contract,
        conviction=0.5,
        primary_reasons=["ml signal said so"],
        # The LLM tried to cite the ml_signal id as if it were a tcid.
        citations=[
            Citation(
                tool_call_id="ml-direction-v0",
                provider_profile_id="yfinance_research",
            ),
        ],
    )
    ttl_table = {
        "price": {"rth": 10, "after_hours": 300, "critical": True},
        "options_chain": {
            "rth_low_vol": 30,
            "rth_high_vol_or_near_expiry": 15,
            "after_hours": 300,
            "critical": True,
        },
    }
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[real_env],
        llm_tool_input={"tool_call_ids_used": ["ml-direction-v0"]},
        registry=registry,
        ttl_table=ttl_table,
        rendered_output=render_template(v, [real_env]),
    )
    assert outcome.skip_reason is SkipReason.phantom_citation
    assert outcome.final_verdict.action is VerdictAction.skip
