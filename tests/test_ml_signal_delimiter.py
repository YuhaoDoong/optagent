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
