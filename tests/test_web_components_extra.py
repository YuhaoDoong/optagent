"""Tests for the v0.4-R2 viz helpers: iv_smile_frame, ledger_index."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from optagent.web.components import iv_smile_frame, ledger_index


def test_iv_smile_frame_filters_bad_iv():
    rows = [
        {"strike": 100, "iv": 0.30, "right": "call"},
        {"strike": 110, "iv": 0.0, "right": "call"},     # filtered (iv too low)
        {"strike": 120, "iv": 6.0, "right": "call"},     # filtered (iv too high)
        {"strike": 0, "iv": 0.30, "right": "put"},       # filtered (strike <= 0)
        {"strike": 100, "iv": "nan", "right": "put"},    # filtered (bad type)
        {"strike": 100, "iv": 0.30, "right": "warrant"}, # filtered (bad right)
        {"strike": 100, "iv": 0.30, "right": "put"},
    ]
    df = iv_smile_frame(rows)
    assert len(df) == 2
    assert set(df["right"].unique()) == {"call", "put"}


def test_iv_smile_frame_sorted_by_strike_per_right():
    rows = [
        {"strike": 100, "iv": 0.30, "right": "call"},
        {"strike": 90, "iv": 0.32, "right": "call"},
        {"strike": 95, "iv": 0.31, "right": "put"},
        {"strike": 105, "iv": 0.29, "right": "put"},
    ]
    df = iv_smile_frame(rows)
    calls = df[df["right"] == "call"]["strike"].tolist()
    puts = df[df["right"] == "put"]["strike"].tolist()
    assert calls == sorted(calls)
    assert puts == sorted(puts)


def test_iv_smile_frame_empty_input():
    assert iv_smile_frame([]).empty


def _write_ledger_row(path: Path, ticker: str, action: str, started_at: datetime) -> None:
    row = {
        "run_id": f"run-{ticker}",
        "ticker": ticker,
        "user_prefs": {},
        "run_mode": "personal_research",
        "envelopes": [],
        "screener_input": {},
        "screener_output": [],
        "prompt_version": "v0",
        "final_verdict": {
            "disclaimer": "x",
            "action": action,
            "skip_reason": "no_candidates_after_screen" if action == "SKIP" else None,
            "primary_reasons": [],
            "dissenting_factors": [],
            "citations": [],
        },
        "validator_decisions": [],
        "unavailable_data_warnings": [],
        "profile_versions": {},
        "started_at": started_at.isoformat(),
        "finished_at": (started_at + timedelta(seconds=1)).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row))
        f.write("\n")


def test_ledger_index_returns_recent_runs(tmp_path: Path):
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    file_today = tmp_path / f"{today.date().isoformat()}.jsonl"
    file_yesterday = tmp_path / f"{yesterday.date().isoformat()}.jsonl"
    _write_ledger_row(file_today, "AAPL", "SKIP", today)
    _write_ledger_row(file_today, "NVDA", "LONG_CALL", today)
    _write_ledger_row(file_yesterday, "SPY", "SKIP", yesterday)

    df = ledger_index(tmp_path, days_back=7)
    assert len(df) == 3
    assert set(df["ticker"]) == {"AAPL", "NVDA", "SPY"}
    # Sorted by started_at DESC
    assert df.iloc[0]["started_at"] >= df.iloc[-1]["started_at"]


def test_ledger_index_skips_old_files(tmp_path: Path):
    today = datetime.now(timezone.utc)
    ancient = today - timedelta(days=30)
    _write_ledger_row(
        tmp_path / f"{ancient.date().isoformat()}.jsonl", "OLD", "SKIP", ancient
    )
    df = ledger_index(tmp_path, days_back=7)
    assert df.empty


def test_ledger_index_handles_missing_directory(tmp_path: Path):
    df = ledger_index(tmp_path / "does_not_exist", days_back=7)
    assert df.empty


def test_ledger_index_skips_malformed_lines(tmp_path: Path):
    today = datetime.now(timezone.utc)
    file_today = tmp_path / f"{today.date().isoformat()}.jsonl"
    file_today.parent.mkdir(parents=True, exist_ok=True)
    with file_today.open("a", encoding="utf-8") as f:
        f.write("{this is not json}\n")
        f.write("\n")  # blank
    _write_ledger_row(file_today, "AAPL", "SKIP", today)

    df = ledger_index(tmp_path, days_back=7)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
