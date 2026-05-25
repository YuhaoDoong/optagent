from __future__ import annotations

import math
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optagent.ml import MLCacheError, MLDirectionAdapter, MLDirectionSignal, RETRAIN_DAYS
from optagent.ml.direction_model import MODEL_VERSION
from optagent.ml.features import FEATURE_NAMES, build_features, build_target


def _synthetic_ohlcv(n: int = 500, drift: float = 0.0005, seed: int = 7) -> pd.DataFrame:
    """Random-walk OHLCV with a small drift, enough rows to train."""

    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1_000_000, 50_000_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


class _FakeYF:
    def __init__(self, history: pd.DataFrame) -> None:
        self.history = history

    def Ticker(self, ticker: str):  # noqa: N802
        df = self.history

        class _T:
            def history(self_inner, *, period="2y", interval="1d", auto_adjust=False):
                return df

        return _T()


# ---------------------------------------------------------------------------
# Feature builder


def test_build_features_returns_all_named_columns():
    df = _synthetic_ohlcv(120)
    f = build_features(df)
    assert tuple(f.columns) == FEATURE_NAMES
    # The last row should be fully populated (all rolling windows filled).
    assert not f.iloc[-1].isna().any()


def test_build_features_rejects_missing_columns():
    df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2]})
    with pytest.raises(ValueError):
        build_features(df)


def test_build_target_shape_and_horizon():
    df = _synthetic_ohlcv(50)
    t = build_target(df["Close"], horizon=5)
    assert len(t) == len(df)
    assert t.iloc[-1] != t.iloc[-1]  # NaN
    # Earlier rows have a defined 0/1 target.
    assert t.iloc[20] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Adapter


def test_signal_unsafe_ticker_returns_none(tmp_path: Path):
    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=None)
    assert ad.signal("../evil") is None
    assert ad.signal("lower-case-bad") is None
    assert ad.signal("") is None


def test_signal_no_yfinance_returns_none(tmp_path: Path):
    """When yfinance is unavailable, signal() must degrade to None."""

    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=None)
    # The constructor may have picked up the real yfinance from the env.
    # Force the "unavailable" state explicitly so the test runs in any env.
    ad._yf = None
    assert ad.signal("AAPL") is None


def test_signal_trains_and_caches(tmp_path: Path):
    yf = _FakeYF(_synthetic_ohlcv(500))
    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=yf)
    sig = ad.signal("AAPL")
    assert isinstance(sig, MLDirectionSignal)
    assert sig.ticker == "AAPL"
    assert 0.0 <= sig.prob_up <= 1.0
    assert sig.class_label in {"up", "down", "neutral"}
    assert sig.model_version == MODEL_VERSION
    assert (tmp_path / "AAPL.pkl").exists()


def test_signal_cache_hit_skips_retrain(tmp_path: Path, monkeypatch):
    yf = _FakeYF(_synthetic_ohlcv(500))
    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=yf)
    sig1 = ad.signal("AAPL")
    assert sig1 is not None
    cache_path = tmp_path / "AAPL.pkl"
    blob = pickle.loads(cache_path.read_bytes())
    blob_trained_at = blob["trained_at"]

    # Second call within retrain window must NOT retrain. We can verify by
    # asserting the cached `trained_at` is unchanged after another signal().
    sig2 = ad.signal("AAPL")
    assert sig2 is not None
    blob_after = pickle.loads(cache_path.read_bytes())
    assert blob_after["trained_at"] == blob_trained_at


def test_signal_stale_cache_triggers_retrain(tmp_path: Path):
    yf = _FakeYF(_synthetic_ohlcv(500))
    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=yf)
    ad.signal("AAPL")
    cache_path = tmp_path / "AAPL.pkl"
    blob = pickle.loads(cache_path.read_bytes())
    # Backdate the trained_at past the retrain window.
    blob["trained_at"] = datetime.now(timezone.utc) - timedelta(days=RETRAIN_DAYS + 1)
    cache_path.write_bytes(pickle.dumps(blob))

    sig = ad.signal("AAPL")
    assert sig is not None
    fresh = pickle.loads(cache_path.read_bytes())
    assert fresh["trained_at"] > blob["trained_at"]


def test_signal_corrupted_cache_falls_back_to_retrain(tmp_path: Path):
    yf = _FakeYF(_synthetic_ohlcv(500))
    cache_path = tmp_path / "AAPL.pkl"
    cache_path.write_bytes(b"definitely not a pickle")
    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=yf)
    sig = ad.signal("AAPL")
    assert sig is not None  # adapter must not raise on corrupted cache


def test_signal_too_little_history_returns_none(tmp_path: Path):
    yf = _FakeYF(_synthetic_ohlcv(120))  # below MIN_TRAIN_ROWS=250
    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=yf)
    assert ad.signal("AAPL") is None


def test_signal_feature_snapshot_is_finite(tmp_path: Path):
    yf = _FakeYF(_synthetic_ohlcv(500))
    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=yf)
    sig = ad.signal("AAPL")
    assert sig is not None
    for k, v in sig.feature_snapshot.items():
        assert math.isfinite(v), f"{k} not finite: {v}"
