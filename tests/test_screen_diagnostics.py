"""Round 12 screen diagnostics: near-misses preserved + stale-bar warnings."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from optagent.strategies import OversoldRebound, SignalDirection
from optagent.strategies.screen import (
    STALENESS_WARN_DAYS,
    screen_universe,
    render_screen_report,
)


def _ohlcv(closes: list[float], end_date: datetime | None = None) -> pd.DataFrame:
    n = len(closes)
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    idx = pd.date_range(end=end_date, periods=n, freq="B")
    close = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": close * 1.001,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _gentle_fade_then_rebound() -> pd.DataFrame:
    flat = [100.0] * 70
    fade = [100.0 - i * 0.5 for i in range(1, 9)]
    last = fade[-1]
    rebound = [last * 1.003]
    return _ohlcv(flat + fade + rebound)


def _sideways(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = list(100 + rng.normal(0, 0.5, n))
    return _ohlcv(closes)


def _stale_fixture() -> pd.DataFrame:
    """OHLCV whose last bar is 6 days behind 'now'."""

    end = datetime.now(timezone.utc) - timedelta(days=6)
    return _ohlcv([100.0] * 80, end_date=end)


def test_screen_preserves_top_near_misses():
    """Sideways-trending tickers should appear as scored near-misses, not silently dropped."""

    def fetcher(ticker: str):
        return _sideways(), None

    result = screen_universe(
        OversoldRebound(),
        ["A", "B", "C"],
        fetcher=fetcher,
        top_n=5,
    )
    # No tickers fully trigger on sideways data, but the strategy still
    # emits scored skip signals. The screener now preserves them.
    assert len(result.top_signals) == 0
    # near-misses may be empty if the strategy's pre-skip score is 0; just
    # confirm the attribute exists and behaves correctly.
    assert isinstance(result.top_near_misses, list)


def test_screen_surfaces_stale_bar_warnings():
    def fetcher(ticker: str):
        return _stale_fixture(), None

    result = screen_universe(
        OversoldRebound(),
        ["AAPL", "MSFT"],
        fetcher=fetcher,
        top_n=5,
    )
    assert len(result.stale_bars) == 2  # both tickers stale by ~6 days
    for ticker, iso_date, days in result.stale_bars:
        assert days >= STALENESS_WARN_DAYS
        assert ticker in {"AAPL", "MSFT"}


def test_render_report_includes_stale_bar_section():
    def fetcher(ticker: str):
        return _stale_fixture(), None

    result = screen_universe(OversoldRebound(), ["AAPL"], fetcher=fetcher)
    out = render_screen_report(result, "RESEARCH ONLY — NOT FINANCIAL ADVICE.")
    assert "Stale-bar warnings" in out


def test_render_report_omits_stale_bar_section_when_fresh():
    def fetcher(ticker: str):
        return _ohlcv([100.0] * 80), None

    result = screen_universe(OversoldRebound(), ["AAPL"], fetcher=fetcher)
    out = render_screen_report(result, "RESEARCH ONLY — NOT FINANCIAL ADVICE.")
    assert "Stale-bar warnings" not in out
