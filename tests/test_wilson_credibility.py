"""Tests for the Codex R4 hardening: Wilson CI on OOS accuracy + class
baseline + n_oos_samples + tighter credibility threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optagent.ml.walk_forward import _wilson_ci, walk_forward_eval


def test_wilson_ci_known_value():
    # 60 successes out of 100 trials at 95% — well-known Wilson lower ~0.503.
    lo, hi = _wilson_ci(0.60, 100)
    assert 0.50 < lo < 0.52
    assert 0.68 < hi < 0.70


def test_wilson_ci_n_zero_returns_full_range():
    lo, hi = _wilson_ci(0.5, 0)
    assert lo == 0.0
    assert hi == 1.0


def test_wilson_ci_clamps_to_unit_interval():
    lo, hi = _wilson_ci(1.0, 5)
    assert lo >= 0.0
    assert hi <= 1.0
    lo2, hi2 = _wilson_ci(0.0, 5)
    assert lo2 >= 0.0
    assert hi2 <= 1.0


def _synthetic(n: int = 800, seed: int = 7, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * 1.005
    low = close * 0.995
    open_ = close * 1.001
    vol = rng.integers(1_000_000, 50_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_walk_forward_carries_ci_and_baseline():
    r = walk_forward_eval(_synthetic())
    assert r is not None
    assert r.n_oos_samples > 0
    assert 0.0 <= r.wilson_ci_lower <= r.wilson_ci_upper <= 1.0
    assert 0.0 <= r.class_baseline_accuracy <= 1.0


def test_walk_forward_dict_includes_new_keys():
    r = walk_forward_eval(_synthetic())
    assert r is not None
    d = r.to_dict()
    for key in (
        "wilson_ci_lower",
        "wilson_ci_upper",
        "n_oos_samples",
        "class_baseline_accuracy",
    ):
        assert key in d


def test_credibility_low_unless_high_evidence(tmp_path):
    """A small/marginal OOS run must NOT get promoted to 'medium' credibility."""

    from optagent.ml.direction_model import MLDirectionAdapter
    from datetime import datetime, timezone

    yf = _FakeYF(_synthetic(n=500))
    ad = MLDirectionAdapter(cache_dir=tmp_path, yf_module=yf)
    sig = ad.signal("AAPL")
    assert sig is not None
    # On synthetic data with a tiny drift, OOS accuracy is rarely high enough
    # to satisfy ci_lower > 0.50 + 0.02 over baseline. credibility = "low".
    assert sig.credibility == "low"


class _FakeYF:
    def __init__(self, history: pd.DataFrame) -> None:
        self._history = history

    def Ticker(self_inner, ticker: str):  # noqa: N802
        df = self_inner._history

        class _T:
            def history(self_inner_t, *, period="2y", interval="1d", auto_adjust=False):
                return df

        return _T()
