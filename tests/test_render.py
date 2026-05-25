from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from optagent import DISCLAIMER
from optagent.render import (
    UnsupportedVerdictError,
    assert_supported_action,
    render_template,
    render_with_disclaimer,
)
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


UTC_NOW = datetime.now(timezone.utc)


def _envelope() -> Envelope:
    return Envelope(
        value={"last": 190.5},
        as_of=UTC_NOW,
        source="yfinance",
        delay_assumption="delayed_15min",
        market_session=MarketSession.rth,
        confidence=Confidence.ok,
        provider_profile_id="yfinance_research",
    )


def _contract() -> OptionContract:
    return OptionContract(
        occ_symbol="AAPL260619C00200000",
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


def test_render_includes_disclaimer_as_first_non_blank_line():
    out = render_with_disclaimer("hello")
    first = out.lstrip().splitlines()[0]
    assert first == DISCLAIMER


def test_render_skip_has_no_contract_section():
    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
        primary_reasons=["Screener empty"],
    )
    memo = render_template(v, [_envelope()])
    assert "Verdict: SKIP" in memo
    assert "Contract:" not in memo


def test_render_long_call_shows_contract_block():
    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.long_call,
        contract=_contract(),
        citations=[Citation(tool_call_id="tc-1", provider_profile_id="yfinance_research")],
    )
    memo = render_template(v, [_envelope()])
    assert "Verdict: LONG_CALL" in memo
    assert "AAPL260619C00200000" in memo
    assert "Breakeven:" in memo
    assert "Max-loss:" in memo


def test_render_fred_attribution_present_iff_cited():
    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
    )
    memo_no = render_template(v, [_envelope()], cited_fred=False)
    assert "Federal Reserve Bank of St. Louis" not in memo_no
    memo_yes = render_template(v, [_envelope()], cited_fred=True)
    assert "Federal Reserve Bank of St. Louis" in memo_yes


def test_render_volume_oi_caveat_present_iff_cited():
    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
    )
    assert "holder cost-basis" not in render_template(v, [_envelope()])
    out = render_template(v, [_envelope()], cited_volume_oi_context=True)
    assert "holder cost-basis" in out


def test_assert_supported_action_accepts_v1_enum():
    assert assert_supported_action("LONG_CALL") is VerdictAction.long_call
    assert assert_supported_action("LONG_PUT") is VerdictAction.long_put
    assert assert_supported_action("SKIP") is VerdictAction.skip


def test_assert_supported_action_rejects_short_and_spreads():
    for bad in ("SHORT_CALL", "SHORT_PUT", "IRON_CONDOR", "NAKED_PUT", "STRADDLE"):
        with pytest.raises(UnsupportedVerdictError):
            assert_supported_action(bad)
