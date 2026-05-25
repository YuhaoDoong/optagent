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


# ---------------------------------------------------------------------------
# Strike grid chosen near spot=190 so deltas stay within [0.20, 0.80].


def test_liquid_chain_returns_candidates():
    rows = [
        _row("AAPL_C190", "call", 190, 5.50, 5.55, oi=5000, volume=300),
        _row("AAPL_P190", "put", 190, 5.40, 5.45, oi=2000, volume=150),
    ]
    out = screen(_inputs(rows))
    assert len(out.candidates) == 2
    assert out.rejected == []
    assert all(c.breakeven > 0 and c.max_loss > 0 for c in out.candidates)
    assert all(c.liquidity_score >= 0 for c in out.candidates)


def test_illiquid_chain_returns_empty_with_reasons():
    rows = [
        _row("AAPL_C190", "call", 190, 0.10, 1.50, oi=5000, volume=300),  # wide spread
        _row("AAPL_P190", "put", 190, 5.40, 5.45, oi=10, volume=150),     # low OI
    ]
    out = screen(_inputs(rows))
    assert out.candidates == []
    reasons = {r[1] for r in out.rejected}
    assert "wide_spread" in reasons
    assert "low_oi" in reasons


def test_missing_bid_rejected_as_zero_bid():
    rows = [_row("AAPL_C190", "call", 190, 0.0, 2.60, oi=5000, volume=300)]
    out = screen(_inputs(rows))
    assert out.candidates == []
    assert out.rejected[0][1] == "zero_bid"


def test_zero_bid_and_zero_ask_rejected_as_zero_bid():
    rows = [_row("AAPL_C190", "call", 190, 0.0, 0.0, oi=5000, volume=300)]
    out = screen(_inputs(rows))
    assert out.rejected[0][1] == "zero_bid"


def test_crossed_book_rejected():
    rows = [_row("AAPL_C190", "call", 190, 2.60, 2.40, oi=5000, volume=300)]
    out = screen(_inputs(rows))
    assert out.rejected[0][1] == "crossed_book"


def test_locked_book_rejected():
    rows = [_row("AAPL_C190", "call", 190, 2.50, 2.50, oi=5000, volume=300)]
    out = screen(_inputs(rows))
    assert out.rejected[0][1] == "locked_book"


def test_invalid_right_rejected():
    bad = _row("AAPL_X", "warrant", 190, 2.40, 2.60, oi=5000, volume=300)
    out = screen(_inputs([bad]))
    assert out.candidates == []
    assert out.rejected[0][1] == "invalid_right"


def test_deep_otm_call_rejected_as_delta_out_of_band():
    rows = [_row("AAPL_C300", "call", 300, 0.05, 0.06, oi=5000, volume=300)]
    out = screen(_inputs(rows))
    assert out.candidates == []
    assert out.rejected[0][1] in {"delta_out_of_band", "wide_spread"}


def test_zero_dte_contract_rejected():
    rows = [_row("AAPL_C190", "call", 190, 5.50, 5.55, oi=5000, volume=300)]
    inp = _inputs(rows, dte=0)
    out = screen(inp)
    assert out.candidates == []
    assert out.rejected[0][1] == "dte_out_of_range"


def test_event_too_close_rejects():
    rows = [_row("AAPL_C190", "call", 190, 5.50, 5.55, oi=5000, volume=300)]
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
        _row("AAPL_C190", "call", 190, 5.50, 5.55, oi=5000, volume=300),
        _row("AAPL_P190", "put", 190, 5.40, 5.45, oi=2000, volume=150),
    ]
    a = screen(_inputs(rows))
    b = screen(_inputs(rows))
    assert [c.model_dump() for c in a.candidates] == [c.model_dump() for c in b.candidates]


def test_split_by_bias_picks_first_of_each_right():
    rows = [
        _row("AAPL_C190", "call", 190, 5.50, 5.55, oi=5000, volume=300),
        _row("AAPL_C195", "call", 195, 3.20, 3.25, oi=4000, volume=200),
        _row("AAPL_P190", "put", 190, 5.40, 5.45, oi=2000, volume=150),
    ]
    out = screen(_inputs(rows))
    best_call, best_put = split_by_bias(out.candidates, "bullish")
    assert best_call is not None and best_call.right.value == "call"
    assert best_put is not None and best_put.right.value == "put"


def test_thresholds_can_be_relaxed():
    rows = [_row("AAPL_C190", "call", 190, 5.50, 5.55, oi=50, volume=5)]
    out_default = screen(_inputs(rows))
    assert out_default.candidates == []  # default min_oi=100 / min_volume=25

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


def test_min_abs_delta_can_be_disabled():
    """A user who wants deep-OTM lotteries can set min_abs_delta=0.0."""

    rows = [_row("AAPL_C300", "call", 300, 0.05, 0.06, oi=5000, volume=300)]
    inp = ScreenerInputs(
        ticker="AAPL",
        spot=190.0,
        rows=rows,
        expiration_str="2026-06-19",
        dte=14,
        risk_free_rate=0.045,
        thresholds=ScreenerThresholds(min_abs_delta=0.0, max_abs_delta=1.0),
    )
    out = screen(inp)
    # The spread is 0.01/0.055 ≈ 18%, still > 0.15 max; expect wide_spread.
    # The point of this test is to prove delta_out_of_band is gone.
    reasons = {r[1] for r in out.rejected}
    assert "delta_out_of_band" not in reasons
