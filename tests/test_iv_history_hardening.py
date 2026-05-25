"""Regression tests for the iv_history hardening (Codex R3 task30 findings)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from optagent.iv_history import (
    IVHistoryError,
    _path_for,
    _safe_ticker,
    append_snapshot,
    compute_iv_rank,
    read_history,
)


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "AAPL/extra",
        "AAPL..",
        "AAPL.",
        "AAPL-",
        "AAPL--B",
        "",
        "x" * 17,  # too long
        "1AAPL",  # must start with letter
    ],
)
def test_safe_ticker_rejects_unsafe(bad):
    with pytest.raises(IVHistoryError):
        _safe_ticker(bad)


def test_safe_ticker_accepts_common_symbols():
    for good in ("AAPL", "SPY", "QQQ", "BRK.B", "BF-B"):
        assert _safe_ticker(good) == good.upper()


def test_path_for_blocks_traversal(tmp_path: Path):
    with pytest.raises(IVHistoryError):
        _path_for("../evil", base=tmp_path)


def test_append_rejects_unsafe_ticker(tmp_path: Path):
    assert append_snapshot("../evil", atm_iv_median=0.25, hv20_annual=0.18, base=tmp_path) is None


def test_append_rejects_future_as_of(tmp_path: Path):
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    assert (
        append_snapshot(
            "AAPL",
            atm_iv_median=0.25,
            hv20_annual=0.18,
            as_of=future,
            base=tmp_path,
        )
        is None
    )


def test_append_accepts_small_skew(tmp_path: Path):
    skewed = datetime.now(timezone.utc) + timedelta(seconds=30)
    path = append_snapshot("AAPL", atm_iv_median=0.25, hv20_annual=0.18, as_of=skewed, base=tmp_path)
    assert path is not None


def test_append_stores_none_for_negative_hv20(tmp_path: Path):
    append_snapshot("AAPL", atm_iv_median=0.25, hv20_annual=-0.5, base=tmp_path)
    history = read_history("AAPL", base=tmp_path)
    assert len(history) == 1
    assert history[0].hv20_annual is None


def test_read_history_filters_future_dated_rows(tmp_path: Path):
    """A manually injected future-dated row must be silently ignored on read."""

    base = tmp_path
    base.mkdir(parents=True, exist_ok=True)
    file = base / "AAPL.jsonl"
    future_iso = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
    today_iso = datetime.now(timezone.utc).isoformat()
    file.write_text(
        "\n".join(
            [
                f'{{"ticker":"AAPL","as_of":"{today_iso}","atm_iv_median":0.25,"hv20_annual":0.18}}',
                f'{{"ticker":"AAPL","as_of":"{future_iso}","atm_iv_median":99.0,"hv20_annual":99.0}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    history = read_history("AAPL", base=base)
    assert len(history) == 1  # poisoned row dropped
    assert history[0].atm_iv_median == 0.25


def test_compute_iv_rank_with_unsafe_ticker_returns_none(tmp_path: Path):
    # Unsafe ticker resolves through _path_for(); read_history must degrade
    # to an empty list rather than raise.
    assert compute_iv_rank("../evil", current_iv=0.25, base=tmp_path) is None
