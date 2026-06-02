"""The empty-screen SKIP must distinguish a data problem from a real no-find."""

from __future__ import annotations

from datetime import datetime, timezone

from optagent.orchestrator import _diagnose_empty_screen
from optagent.render import _market_snapshot_lines, render_template
from optagent.schemas import (
    Confidence,
    Envelope,
    MarketSession,
    SkipReason,
    Verdict,
    VerdictAction,
)
from optagent.screener import ScreenerOutput


def _screen_output(rejected, n_rows):
    return ScreenerOutput(
        candidates=[],
        rejected=rejected,
        inputs_summary={"n_rows_in": n_rows, "expiration": "2026-06-10"},
    )


def test_quoteless_chain_maps_to_stale_required_input():
    rejected = [(f"OCC{i}", "zero_bid") for i in range(53)]
    reason, msgs = _diagnose_empty_screen(_screen_output(rejected, 53))
    assert reason is SkipReason.stale_required_input
    joined = " ".join(msgs).lower()
    assert "bid" in joined and "market" in joined
    assert "53/53" in msgs[0]


def test_empty_chain_maps_to_stale_required_input():
    reason, msgs = _diagnose_empty_screen(_screen_output([], 0))
    assert reason is SkipReason.stale_required_input
    assert "empty" in " ".join(msgs).lower()


def test_real_filter_miss_keeps_no_candidates_reason():
    # Quotes were fine; nothing met the liquidity / delta bar.
    rejected = (
        [(f"A{i}", "low_oi") for i in range(6)]
        + [(f"B{i}", "delta_out_of_band") for i in range(4)]
    )
    reason, msgs = _diagnose_empty_screen(_screen_output(rejected, 10))
    assert reason is SkipReason.no_candidates_after_screen
    assert "liquidity" in " ".join(msgs).lower()


def test_mixed_but_mostly_quoteless_is_data_problem():
    rejected = [(f"A{i}", "zero_bid") for i in range(9)] + [("B0", "low_oi")]
    reason, _ = _diagnose_empty_screen(_screen_output(rejected, 10))
    assert reason is SkipReason.stale_required_input


# --- market snapshot ---------------------------------------------------------


def _env(value, source="yfinance", profile="yfinance_research"):
    return Envelope(
        value=value,
        as_of=datetime(2026, 6, 2, tzinfo=timezone.utc),
        source=source,
        delay_assumption="delayed_15min",
        market_session=MarketSession.closed,
        confidence=Confidence.ok,
        provider_profile_id=profile,
    )


def test_market_snapshot_extracts_known_fields():
    envs = [
        _env({"ticker": "AAPL", "last": 306.31}),
        _env({"last_close": 306.31, "recent_high_60d": 312.5, "recent_low_60d": 246.6, "hv20_annual": 0.169}),
        _env(
            {"next_event": {"date": "2026-06-06", "kind": "NFP", "label": "Nonfarm payrolls"}, "days_to_next_event": 4},
            source="econ_calendar_builtin",
            profile="econ_calendar_builtin",
        ),
    ]
    lines = _market_snapshot_lines(envs)
    blob = "\n".join(lines)
    assert "306.31" in blob
    assert "60-day range" in blob
    assert "16.9%" in blob
    assert "Nonfarm payrolls" in blob and "in 4d" in blob


def test_skip_memo_includes_snapshot_and_keeps_no_contract_section():
    v = Verdict(
        disclaimer="RESEARCH ONLY.",
        action=VerdictAction.skip,
        skip_reason=SkipReason.stale_required_input,
        primary_reasons=["no live quotes"],
    )
    memo = render_template(v, [_env({"ticker": "AAPL", "last": 306.31})])
    assert "Verdict: SKIP" in memo
    assert "Market snapshot" in memo
    assert "Contract:" not in memo  # SKIP must never render a contract block


def test_market_snapshot_empty_when_no_known_fields():
    assert _market_snapshot_lines([_env({"unrelated": 1})]) == []
