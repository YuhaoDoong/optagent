"""Regression tests for the Codex R4 hardening (Round 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optagent import DISCLAIMER
from optagent.ml.walk_forward import walk_forward_eval
from optagent.strategies.base import (
    DiagnosticBlock,
    SignalDirection,
    StrategySignal,
)
from optagent.strategies.screen import ScreenResult, screen_universe
from optagent.strategies.oversold_rebound import OversoldRebound


def _synthetic(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0005, 0.012, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * 1.005
    low = close * 0.995
    open_ = close * 1.001
    vol = rng.integers(1_000_000, 50_000_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_walk_forward_rejects_gap_below_horizon():
    """Codex R4: gap < horizon enables label leakage; must raise."""

    df = _synthetic()
    with pytest.raises(ValueError):
        walk_forward_eval(df, gap=2, horizon=5)


def test_walk_forward_accepts_gap_equal_horizon():
    """Equal is fine — last train label ends just before the val window."""

    df = _synthetic()
    r = walk_forward_eval(df, gap=5, horizon=5)
    assert r is not None


def test_strategy_signal_disclaimer_is_canonical_even_if_overridden():
    """Codex R4: __post_init__ must overwrite any non-canonical disclaimer."""

    sig = StrategySignal(
        strategy_id="oversold_rebound",
        ticker="TEST",
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        spot=100.0,
        direction=SignalDirection.skip,
        score=0.0,
        daily=DiagnosticBlock(label="daily", summary="x", conditions={}),
        disclaimer="LIES — this is real advice",
    )
    assert sig.disclaimer == DISCLAIMER


def test_screen_universe_survives_fetcher_exceptions():
    """Codex R4: one bad-fetch ticker must not kill the screen."""

    def fetcher(ticker: str):
        if ticker == "BAD":
            raise RuntimeError("network is broken for this ticker")
        if ticker == "GOOD":
            return _synthetic(80), None
        return None, None

    result = screen_universe(
        OversoldRebound(),
        ["BAD", "GOOD", "MISSING"],
        fetcher=fetcher,
        top_n=5,
    )
    # `BAD` must appear in skipped with a fetcher_raised reason.
    reasons = dict(result.skipped)
    assert "BAD" in reasons
    assert reasons["BAD"].startswith("fetcher_raised:")
    # The screen still completed instead of bubbling the exception.
    assert isinstance(result, ScreenResult)


def test_ml_cache_invalidates_on_model_version_mismatch(tmp_path):
    """Codex R4: cache from a different MODEL_VERSION must be ignored."""

    import pickle
    from datetime import datetime, timezone

    from optagent.ml.direction_model import MLDirectionAdapter

    # Pre-seed a cache with a different model_version.
    blob = {
        "model": object(),  # placeholder; never used because we'll re-train
        "trained_at": datetime.now(timezone.utc),
        "ticker": "AAPL",
        "n_train_rows": 100,
        "accuracy_self": 0.99,
        "feature_names": ["wrong", "schema"],
        "model_version": "ml-direction-OLD-VERSION",
    }
    path = tmp_path / "AAPL.pkl"
    path.write_bytes(pickle.dumps(blob))

    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=None)
    # _load() must reject the wrong-version blob.
    assert ad._load(path) is None
