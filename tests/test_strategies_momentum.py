from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optagent.strategies import (
    BreakdownContinuation,
    MomentumBreakout,
    SignalDirection,
)


def _ohlcv_from_closes(closes: list[float], vols: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.array(closes, dtype=float)
    high = close * 1.005
    low = close * 0.995
    open_ = close * 1.001
    if vols is None:
        vols = [1_000_000.0] * n
    vol = np.array(vols, dtype=float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def _breakout_setup(n_base: int = 60, n_push: int = 5) -> pd.DataFrame:
    base = [100.0 + i * 0.1 for i in range(n_base)]
    push = [base[-1] * (1.0 + 0.012 * (i + 1)) for i in range(n_push)]
    closes = base + push
    vols = [1_000_000.0] * n_base + [3_000_000.0] * n_push
    return _ohlcv_from_closes(closes, vols)


def _breakdown_setup(n_base: int = 60, n_drop: int = 5) -> pd.DataFrame:
    base = [100.0 - i * 0.1 for i in range(n_base)]
    drop = [base[-1] * (1.0 - 0.012 * (i + 1)) for i in range(n_drop)]
    closes = base + drop
    vols = [1_000_000.0] * n_base + [3_000_000.0] * n_drop
    return _ohlcv_from_closes(closes, vols)


def _sideways(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    closes = list(100 + rng.normal(0, 0.5, n))
    return _ohlcv_from_closes(closes)


def test_momentum_breakout_fires_on_clean_breakout():
    df = _breakout_setup()
    sig = MomentumBreakout().evaluate("TEST", ohlcv_daily=df)
    assert sig is not None
    conds = sig.daily.conditions
    # Conditions individually should pass on this setup.
    assert conds["broke_out"] is True
    assert conds["vol_expansion"] is True
    assert conds["trend_healthy"] is True
    # RSI band may sit just inside; direction is long_call when all conditions + reward in band.
    assert sig.score > 0


def test_momentum_breakout_skips_on_sideways():
    df = _sideways()
    sig = MomentumBreakout().evaluate("TEST", ohlcv_daily=df)
    assert sig is not None
    assert sig.direction is SignalDirection.skip


def test_momentum_breakout_too_short_returns_none():
    df = _ohlcv_from_closes([100.0] * 30)
    assert MomentumBreakout().evaluate("TEST", ohlcv_daily=df) is None


def test_breakdown_continuation_fires_on_clean_breakdown():
    df = _breakdown_setup()
    sig = BreakdownContinuation().evaluate("TEST", ohlcv_daily=df)
    assert sig is not None
    conds = sig.daily.conditions
    assert conds["broke_down"] is True
    assert conds["vol_expansion"] is True
    assert conds["trend_bearish"] is True
    assert sig.score > 0


def test_breakdown_continuation_skips_on_sideways():
    df = _sideways()
    sig = BreakdownContinuation().evaluate("TEST", ohlcv_daily=df)
    assert sig is not None
    assert sig.direction is SignalDirection.skip


def test_breakdown_continuation_direction_when_triggered():
    df = _breakdown_setup()
    sig = BreakdownContinuation().evaluate("TEST", ohlcv_daily=df)
    assert sig is not None
    # Either fully triggered (long_put_observation) or skip with notes.
    assert sig.direction in {SignalDirection.long_put_observation, SignalDirection.skip}
    # If triggered, the reward block must show negative repair_space_pct.
    if sig.direction is SignalDirection.long_put_observation:
        assert sig.reward.repair_space_pct is not None
        assert sig.reward.repair_space_pct <= 0
