from __future__ import annotations

from optagent.screener import (
    ScreenerInputs,
    ScreenerThresholds,
    screen,
    split_by_bias,
)


def _row(
    occ: str,
    right: str,
    strike: float,
    bid: float,
    ask: float,
    oi: int,
    volume: int,
    iv: float = 0.25,
) -> dict:
    return {
        "occ_symbol": occ,
        "strike": strike,
        "right": right,
        "bid": bid,
        "ask": ask,
        "last_price": (bid + ask) / 2.0,
        "volume": volume,
        "open_interest": oi,
        "iv": iv,
    }


def _inputs(rows, dte: int = 14) -> ScreenerInputs:
    return ScreenerInputs(
        ticker="AAPL",
        spot=190.0,
        rows=rows,
        expiration_str="2026-06-19",
        dte=dte,
        risk_free_rate=0.045,
        days_to_event=None,
    )


def test_liquid_chain_returns_candidates():
    rows = [
        _row("AAPL_C200", "call", 200, 2.40, 2.60, oi=5000, volume=300),
        _row("AAPL_P180", "put", 180, 1.20, 1.40, oi=2000, volume=150),
    ]
    out = screen(_inputs(rows))
    assert len(out.candidates) == 2
    assert out.rejected == []
    assert all(c.breakeven > 0 and c.max_loss > 0 for c in out.candidates)
    assert all(c.liquidity_score >= 0 for c in out.candidates)


def test_illiquid_chain_returns_empty_with_reasons():
    rows = [
        _row("AAPL_C200", "call", 200, 0.10, 1.50, oi=5000, volume=300),  # wide spread
        _row("AAPL_P180", "put", 180, 1.20, 1.40, oi=10, volume=150),    # low OI
    ]
    out = screen(_inputs(rows))
    assert out.candidates == []
    reasons = {r[1] for r in out.rejected}
    assert "wide_spread" in reasons
    assert "low_oi" in reasons


def test_missing_bid_rejected_with_missing_field():
    rows = [_row("AAPL_C200", "call", 200, 0.0, 2.60, oi=5000, volume=300)]
    out = screen(_inputs(rows))
    assert out.candidates == []
    assert out.rejected[0][1] == "missing_field"


def test_zero_dte_contract_rejected():
    rows = [_row("AAPL_C200", "call", 200, 2.40, 2.60, oi=5000, volume=300)]
    inp = _inputs(rows, dte=0)
    out = screen(inp)
    # Single row, DTE outside window → rejected as dte_out_of_range
    assert out.candidates == []
    assert out.rejected[0][1] == "dte_out_of_range"


def test_event_too_close_rejects():
    rows = [_row("AAPL_C200", "call", 200, 2.40, 2.60, oi=5000, volume=300)]
    inp = ScreenerInputs(
        ticker="AAPL",
        spot=190.0,
        rows=rows,
        expiration_str="2026-06-19",
        dte=14,
        risk_free_rate=0.045,
        days_to_event=0,  # earnings today
    )
    out = screen(inp)
    assert out.candidates == []
    assert out.rejected[0][1] == "event_too_close"


def test_screen_is_deterministic():
    rows = [
        _row("AAPL_C200", "call", 200, 2.40, 2.60, oi=5000, volume=300),
        _row("AAPL_P180", "put", 180, 1.20, 1.40, oi=2000, volume=150),
    ]
    a = screen(_inputs(rows))
    b = screen(_inputs(rows))
    assert [c.model_dump() for c in a.candidates] == [c.model_dump() for c in b.candidates]


def test_split_by_bias_picks_first_of_each_right():
    rows = [
        _row("AAPL_C200", "call", 200, 2.40, 2.60, oi=5000, volume=300),
        _row("AAPL_C210", "call", 210, 1.40, 1.50, oi=4000, volume=200),
        _row("AAPL_P180", "put", 180, 1.20, 1.40, oi=2000, volume=150),
    ]
    out = screen(_inputs(rows))
    best_call, best_put = split_by_bias(out.candidates, "bullish")
    assert best_call is not None and best_call.right.value == "call"
    assert best_put is not None and best_put.right.value == "put"


def test_thresholds_can_be_relaxed():
    rows = [_row("AAPL_C200", "call", 200, 2.40, 2.60, oi=50, volume=5)]
    out_default = screen(_inputs(rows))
    assert out_default.candidates == []  # default min_oi=100 / min_volume=10

    inp = ScreenerInputs(
        ticker="AAPL",
        spot=190.0,
        rows=rows,
        expiration_str="2026-06-19",
        dte=14,
        risk_free_rate=0.045,
        thresholds=ScreenerThresholds(min_oi=10, min_volume=1),
    )
    out_relaxed = screen(inp)
    assert len(out_relaxed.candidates) == 1
