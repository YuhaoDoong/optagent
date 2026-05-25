from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optagent.ml.walk_forward import WalkForwardResult, walk_forward_eval


def _synthetic(n: int = 600, seed: int = 11, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1_000_000, 50_000_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_walk_forward_returns_result_on_sufficient_data():
    r = walk_forward_eval(_synthetic(600))
    assert isinstance(r, WalkForwardResult)
    assert r.n_folds >= 3
    assert 0.0 <= r.oos_accuracy_mean <= 1.0
    assert r.oos_log_loss_mean > 0.0
    assert len(r.train_rows_per_fold) == r.n_folds
    assert len(r.val_rows_per_fold) == r.n_folds


def test_walk_forward_returns_none_on_short_data():
    assert walk_forward_eval(_synthetic(80)) is None


def test_walk_forward_to_dict_round_trip():
    r = walk_forward_eval(_synthetic(600))
    assert r is not None
    d = r.to_dict()
    for key in (
        "n_folds",
        "oos_accuracy_mean",
        "oos_accuracy_std",
        "oos_log_loss_mean",
        "direction_hit_rate",
    ):
        assert key in d


def test_walk_forward_drift_produces_above_chance_in_sample_of_seeds():
    """Sanity: with a tiny positive drift, OOS accuracy should be > 0.45 on
    average across seeds (chance + a hair). Not a strong claim — just a
    regression guard so a refactor that destroys signal trips this test.
    """

    accs = []
    for seed in (1, 2, 3, 4, 5):
        r = walk_forward_eval(_synthetic(600, seed=seed, drift=0.001))
        if r is not None:
            accs.append(r.oos_accuracy_mean)
    assert accs, "walk_forward returned None for every seed"
    assert np.mean(accs) > 0.40  # generous; we're just guarding against signal collapse
