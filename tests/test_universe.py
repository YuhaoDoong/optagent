from __future__ import annotations

from pathlib import Path

import pytest

from optagent.strategies.universe import (
    BUILTIN_US_LARGE_CAP,
    UniverseFilter,
    builtin_us_large_cap,
    load_universe,
)


def test_builtin_universe_has_reasonable_size():
    u = builtin_us_large_cap()
    assert 30 <= len(u) <= 100
    assert "AAPL" in u
    assert "SPY" in u


def test_load_universe_builtin_alias():
    assert load_universe("builtin") == builtin_us_large_cap()


def test_load_universe_from_file(tmp_path: Path):
    p = tmp_path / "u.txt"
    p.write_text("# comment\nAAPL\nMSFT\n\nNVDA\n", encoding="utf-8")
    u = load_universe(p)
    assert u == ["AAPL", "MSFT", "NVDA"]


def test_load_universe_from_iterable():
    assert load_universe(["aapl", "msft"]) == ["AAPL", "MSFT"]


def test_load_universe_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_universe(tmp_path / "does_not_exist.txt")


def test_universe_filter_keeps_tickers_when_lookup_unavailable(tmp_path: Path):
    """If yfinance isn't available, the filter MUST keep tickers (no false rejection)."""

    f = UniverseFilter(
        min_market_cap_usd=1e12,
        min_avg_volume=1e9,
        cache_path=tmp_path / "c.json",
        yf_module=None,
    )
    out = f.apply(["AAPL", "MSFT"])
    assert out == ["AAPL", "MSFT"]


def test_universe_filter_applies_market_cap_threshold(tmp_path: Path):
    class _FakeFastInfo:
        def __init__(self, mc: float, vol: float) -> None:
            self.market_cap = mc
            self.ten_day_average_volume = vol
            self.three_month_average_volume = vol

    class _FakeTicker:
        def __init__(self, mc: float, vol: float) -> None:
            self.fast_info = _FakeFastInfo(mc, vol)

    class _FakeYF:
        def Ticker(self_inner, ticker: str):  # noqa: N802
            return {"BIG": _FakeTicker(3e12, 5e7), "SMALL": _FakeTicker(5e8, 1e5)}[ticker]

    f = UniverseFilter(
        min_market_cap_usd=1e9,
        cache_path=tmp_path / "c.json",
        yf_module=_FakeYF(),
    )
    out = f.apply(["BIG", "SMALL"])
    assert out == ["BIG"]


def test_universe_filter_caches_results(tmp_path: Path):
    calls = {"n": 0}

    class _FakeYF:
        def Ticker(self_inner, ticker: str):  # noqa: N802
            calls["n"] += 1

            class _T:
                class fast_info:
                    market_cap = 2e12
                    ten_day_average_volume = 5e7
                    three_month_average_volume = 5e7

            return _T()

    cache_path = tmp_path / "c.json"
    f1 = UniverseFilter(cache_path=cache_path, yf_module=_FakeYF())
    f1.apply(["AAPL"])
    assert calls["n"] == 1
    # Second filter instance reads the cache instead of re-fetching.
    f2 = UniverseFilter(cache_path=cache_path, yf_module=_FakeYF())
    f2.apply(["AAPL"])
    assert calls["n"] == 1
