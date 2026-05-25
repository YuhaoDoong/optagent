from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from optagent.strategies import (
    OversoldRebound,
    SignalDirection,
    StrategySignal,
)
from optagent.strategies.screen import (
    ScreenResult,
    render_screen_report,
    screen_universe,
)


def _ohlcv(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.array(closes, dtype=float)
    high = close * 1.005
    low = close * 0.995
    open_ = close * 1.001
    vol = np.full(n, 1_000_000.0)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def _flat_then_crash(flat_n: int = 70, crash_n: int = 10, slope: float = 1.2) -> pd.DataFrame:
    flat = [100.0] * flat_n
    crash = [100.0 - i * slope for i in range(1, crash_n + 1)]
    return _ohlcv(flat + crash)


def _gentle_oversold(flat_n: int = 70, fade_n: int = 8, slope: float = 0.5) -> pd.DataFrame:
    """Smooth slow-bleed setup so ATR doesn't explode but RSI/WR/EMA20 dev
    still print oversold readings. Used by the full-trigger regression test.
    """

    flat = [100.0] * flat_n
    fade = [100.0 - i * slope for i in range(1, fade_n + 1)]
    return _ohlcv(flat + fade)


def _strong_uptrend(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.005, 0.005, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    return _ohlcv(list(close))


def test_oversold_fires_on_crash_setup():
    df = _flat_then_crash()
    strategy = OversoldRebound()
    sig = strategy.evaluate("TEST", ohlcv_daily=df)
    assert isinstance(sig, StrategySignal)
    # The 10-day decline should drive RSI down, WR very negative, EMA20 dev < -4.5%,
    # consecutive_down_days well above 3 — strong oversold setup.
    conds = sig.daily.conditions
    assert conds["rsi_14_pass"] is True
    assert conds["williams_r_14_pass"] is True
    assert conds["consec_down_pass"] is True
    # Direction depends also on ATR not expanding + stop_bleed; a fresh crash
    # may not satisfy stop_bleed (latest close < prior close). Allow either
    # SKIP or LONG_CALL but score must be > 0.
    assert sig.score > 0


def test_oversold_fires_on_gentle_fade_with_rebound_bar():
    """Codex R4 regression: the full trigger must be REACHABLE.

    The original logic counted the latest bar in `consec_down` AND required
    `stop_bleed = latest > prior`, which are mutually exclusive. The fix
    counts consec_down on close[:-1] so this scenario triggers.
    """

    df = _gentle_oversold(flat_n=70, fade_n=8, slope=0.5)
    # Append a single rebound bar after the fade.
    last_idx = df.index[-1]
    last_close = float(df["Close"].iloc[-1])
    rebound_close = last_close * 1.003
    new_row = pd.DataFrame(
        {
            "Open": [last_close],
            "High": [rebound_close * 1.001],
            "Low": [last_close * 0.999],
            "Close": [rebound_close],
            "Volume": [1_000_000.0],
        },
        index=[last_idx + pd.tseries.offsets.BDay()],
    )
    df2 = pd.concat([df, new_row])

    sig = OversoldRebound().evaluate("TEST", ohlcv_daily=df2)
    assert sig is not None
    conds = sig.daily.conditions
    assert conds["consec_down_pass"] is True, "lead-in down-run must count after fix"
    # The intraday block uses latest > prior as the stop_bleed proxy.
    assert sig.intraday is not None
    assert sig.intraday.conditions["stop_bleed"] is True, "stop_bleed must be reachable after fix"
    # The exact direction depends on ATR / repair-band — but the two
    # CRITICAL conditions (consec_down AND stop_bleed) are now both True,
    # which the pre-fix code made impossible.


def test_oversold_does_not_fire_on_uptrend():
    df = _strong_uptrend()
    strategy = OversoldRebound()
    sig = strategy.evaluate("TEST", ohlcv_daily=df)
    assert sig is not None
    assert sig.direction is SignalDirection.skip
    conds = sig.daily.conditions
    assert conds["rsi_14_pass"] is False
    assert conds["williams_r_14_pass"] is False


def test_oversold_returns_none_on_insufficient_history():
    sig = OversoldRebound().evaluate("TEST", ohlcv_daily=_ohlcv([100.0] * 30))
    assert sig is None


def test_strategy_signal_to_dict_is_serialisable():
    df = _flat_then_crash()
    sig = OversoldRebound().evaluate("TEST", ohlcv_daily=df)
    assert sig is not None
    d = sig.to_dict()
    import json

    assert json.dumps(d)  # round-trips without error
    assert d["disclaimer"].startswith("RESEARCH ONLY")


def test_screen_universe_picks_top_by_score():
    def fetcher(ticker: str):
        if ticker == "CRASH":
            return _flat_then_crash(), None
        if ticker == "UP":
            return _strong_uptrend(), None
        return None, None

    result = screen_universe(
        OversoldRebound(),
        ["CRASH", "UP", "MISSING"],
        fetcher=fetcher,
        top_n=2,
    )
    assert isinstance(result, ScreenResult)
    assert result.universe_size == 3
    assert result.n_evaluated == 2
    # CRASH may or may not trigger depending on stop_bleed; UP must NOT.
    triggered_tickers = [s.ticker for s in result.top_signals]
    assert "UP" not in triggered_tickers
    # MISSING ticker should appear in `skipped`.
    skip_tickers = [t for (t, _) in result.skipped]
    assert "MISSING" in skip_tickers


def test_render_screen_report_starts_with_disclaimer():
    result = screen_universe(
        OversoldRebound(),
        [],  # empty universe
        fetcher=lambda t: (None, None),
        top_n=5,
    )
    out = render_screen_report(result, disclaimer="RESEARCH ONLY — NOT FINANCIAL ADVICE.")
    assert out.splitlines()[0] == "RESEARCH ONLY — NOT FINANCIAL ADVICE."
    assert "Top candidates" in out
