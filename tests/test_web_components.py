"""Unit tests for the Streamlit-free UI helpers.

We test the pure helpers in `optagent.web.components` because they're
the part that needs to be reliable (Streamlit page bodies are
notoriously hard to unit-test). When v0.4 ports to FastAPI, these
helpers move over unchanged.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from optagent import DISCLAIMER
from optagent.schemas import (
    Citation,
    Confidence,
    Envelope,
    MarketSession,
    OptionContract,
    OptionRight,
    SkipReason,
    Verdict,
    VerdictAction,
)
from optagent.web.components import (
    DISCLAIMER_BANNER,
    candidate_table,
    candle_chart,
    envelope_summary,
    feature_radar,
    ml_signal_gauge,
    strategy_signal_table,
    verdict_badge,
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
        warnings=["x"],
    )


def _contract(occ: str = "AAPL_C200") -> OptionContract:
    return OptionContract(
        occ_symbol=occ,
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


def test_disclaimer_banner_uses_canonical_constant():
    assert DISCLAIMER_BANNER == DISCLAIMER


def test_verdict_badge_skip():
    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
    )
    out = verdict_badge(v)
    assert out["action"] == "SKIP"
    assert out["color"].startswith("#")
    assert out["skip_reason"] == "no_candidates_after_screen"


def test_verdict_badge_long_call():
    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.long_call,
        contract=_contract(),
        conviction=0.65,
    )
    out = verdict_badge(v)
    assert out["action"] == "LONG_CALL"
    assert out["conviction"] == 0.65
    assert "observation" in out["label"].lower()


def test_candidate_table_has_expected_columns():
    df = candidate_table([_contract(), _contract("AAPL_P200")])
    assert {"OCC", "Right", "Strike", "Mid", "Δ", "θ/day", "Max-loss $"} <= set(df.columns)
    assert len(df) == 2


def test_envelope_summary_basic_shape():
    df = envelope_summary([_envelope(), _envelope()])
    assert {"source", "profile", "confidence", "cache_age_s", "warnings"} <= set(df.columns)
    assert len(df) == 2


def test_ml_signal_gauge_handles_missing_signal():
    out = ml_signal_gauge(None)
    assert out == {"available": False}
    out2 = ml_signal_gauge({})
    assert out2 == {"available": False}


def test_ml_signal_gauge_extracts_subtitle_fields():
    out = ml_signal_gauge(
        {
            "prob_up": 0.71,
            "class_label": "up",
            "credibility": "low",
            "oos_accuracy": 0.52,
            "wilson_ci_lower": 0.45,
            "wilson_ci_upper": 0.59,
            "class_baseline_accuracy": 0.53,
            "n_oos_samples": 200,
            "feature_snapshot": {"rsi_14": 60.0},
        }
    )
    assert out["available"] is True
    assert out["prob_up"] == 0.71
    assert "credibility" in out["subtitle"]
    assert "CI 95%" in out["subtitle"]


def test_feature_radar_normalises_known_features():
    snap = {
        "rsi_14": 90.0,         # well above 50 → normalised positive, clipped to 1.0
        "ret_20d": 0.06,
        "atr_14_pct": 0.03,
        "vol_change_5d": 1.5,
        "_garbage": float("inf"),
        "_alsobad": "nan",
    }
    df = feature_radar(snap)
    names = set(df["feature"].tolist())
    assert "rsi_14" in names
    assert "_garbage" not in names
    rsi_row = df[df["feature"] == "rsi_14"].iloc[0]
    assert abs(rsi_row["normalised"]) <= 1.0
    assert rsi_row["normalised"] > 0


def test_candle_chart_adds_ema_columns():
    closes = list(range(100, 180))
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        }
    )
    out = candle_chart(df, max_rows=40)
    assert "EMA20" in out.columns
    assert "EMA50" in out.columns
    assert len(out) == 40


def test_candle_chart_empty_input_returns_empty_df():
    out = candle_chart(pd.DataFrame())
    assert out.empty


def test_strategy_signal_table_extracts_fields():
    from optagent.strategies.base import (
        DiagnosticBlock,
        SignalDirection,
        StrategySignal,
    )

    sig = StrategySignal(
        strategy_id="oversold_rebound",
        ticker="AAPL",
        timestamp=UTC_NOW,
        spot=190.0,
        direction=SignalDirection.long_call_observation,
        score=0.7,
        daily=DiagnosticBlock(
            label="daily",
            summary="ok",
            conditions={"rsi_14": 32.0, "williams_r_14": -90.0, "ema20_dev": -0.06},
        ),
        notes=["test"],
    )
    df = strategy_signal_table([sig])
    assert df.iloc[0]["Ticker"] == "AAPL"
    assert df.iloc[0]["Direction"] == "long_call_observation"
    assert df.iloc[0]["RSI"] == 32.0


def test_strategy_signal_table_empty_input_returns_empty():
    df = strategy_signal_table([])
    assert df.empty
