from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from optagent.iv_history import (
    DEFAULT_MIN_OBSERVATIONS,
    IVSnapshot,
    append_snapshot,
    compute_iv_rank,
    median_iv_from_chain_rows,
    read_history,
)


def test_append_and_read_round_trip(tmp_path: Path):
    written = append_snapshot(
        "AAPL",
        atm_iv_median=0.25,
        hv20_annual=0.18,
        base=tmp_path,
    )
    assert written is not None and written.exists()
    history = read_history("AAPL", base=tmp_path)
    assert len(history) == 1
    assert history[0].ticker == "AAPL"
    assert history[0].atm_iv_median == 0.25
    assert history[0].hv20_annual == 0.18


def test_append_rejects_non_finite(tmp_path: Path):
    assert append_snapshot("AAPL", atm_iv_median=float("nan"), hv20_annual=0.18, base=tmp_path) is None
    assert append_snapshot("AAPL", atm_iv_median=-0.1, hv20_annual=0.18, base=tmp_path) is None


def test_read_returns_empty_for_unknown_ticker(tmp_path: Path):
    assert read_history("ZZZZ", base=tmp_path) == []


def test_compute_iv_rank_returns_none_below_min_observations(tmp_path: Path):
    for _ in range(DEFAULT_MIN_OBSERVATIONS - 1):
        append_snapshot("AAPL", atm_iv_median=0.20, hv20_annual=None, base=tmp_path)
    assert compute_iv_rank("AAPL", current_iv=0.25, base=tmp_path) is None


def test_compute_iv_rank_returns_pct_when_window_full(tmp_path: Path):
    base_t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(DEFAULT_MIN_OBSERVATIONS):
        # IV linearly from 0.10 to 0.10 + (n-1)*0.005
        iv = 0.10 + i * 0.005
        append_snapshot("AAPL", atm_iv_median=iv, hv20_annual=None, as_of=base_t + timedelta(days=i), base=tmp_path)
    rank = compute_iv_rank("AAPL", current_iv=0.20, base=tmp_path)
    assert rank is not None
    assert 0.0 <= rank["rank_pct"] <= 100.0
    # current_iv 0.20 sits inside the [0.10, 0.10 + 29*0.005=0.245] range
    assert 60.0 < rank["rank_pct"] < 80.0
    assert rank["n_observations"] == DEFAULT_MIN_OBSERVATIONS


def test_compute_iv_rank_clamps_outliers(tmp_path: Path):
    base_t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(DEFAULT_MIN_OBSERVATIONS):
        append_snapshot("AAPL", atm_iv_median=0.20, hv20_annual=None, as_of=base_t + timedelta(days=i), base=tmp_path)
    # current_iv way above any historical value → clamped to 50 (flat history)
    rank = compute_iv_rank("AAPL", current_iv=10.0, base=tmp_path)
    assert rank is not None
    # When all historical IVs equal 0.20, range is zero -> rank=50 (neutral).
    assert rank["rank_pct"] == 50.0


def test_median_iv_from_chain_rows_ignores_garbage():
    rows = [
        {"iv": 0.25}, {"iv": 0.30}, {"iv": 0.20},
        {"iv": "nan"}, {"iv": -1}, {"iv": 99}, {"iv": None},
    ]
    m = median_iv_from_chain_rows(rows)
    assert m == 0.25  # middle of sorted [0.20, 0.25, 0.30]


def test_median_iv_returns_none_when_no_sane_iv():
    assert median_iv_from_chain_rows([{"iv": -1}, {"iv": "bad"}]) is None
