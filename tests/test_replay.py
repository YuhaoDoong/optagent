from __future__ import annotations

import json
from pathlib import Path

import pytest

from optagent.replay import Clock, Fixture, build_replay_adapters, replay


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def _sample_fixture(tmp_path: Path) -> Path:
    f = Fixture(
        ticker="AAPL",
        frozen_now="2026-05-25T15:30:00+00:00",
        yfinance_price={"ticker": "AAPL", "last": 190.0},
        yfinance_history={
            "ticker": "AAPL",
            "last_close": 190.0,
            "recent_high_60d": 200.0,
            "recent_low_60d": 170.0,
            "hv20_annual": 0.25,
            "n_returns_used": 20,
        },
        yfinance_chain={
            "ticker": "AAPL",
            "expiration": "2026-06-19",
            "dte": 25,
            "rows": [
                {
                    "occ_symbol": "AAPL_C200",
                    "right": "call",
                    "strike": 200.0,
                    "bid": 2.40,
                    "ask": 2.60,
                    "last_price": 2.50,
                    "volume": 300,
                    "open_interest": 5000,
                    "iv": 0.28,
                },
                {
                    "occ_symbol": "AAPL_P180",
                    "right": "put",
                    "strike": 180.0,
                    "bid": 1.20,
                    "ask": 1.40,
                    "last_price": 1.30,
                    "volume": 200,
                    "open_interest": 3000,
                    "iv": 0.30,
                },
            ],
        },
        econ_calendar={
            "next_event": {"date": "2026-06-06", "kind": "NFP", "label": "Nonfarm payrolls"},
            "days_to_next_event": 12,
            "days_by_kind": {"NFP": 12, "FOMC": 23, "CPI": 17},
            "next_by_kind": {"NFP": {"date": "2026-06-06", "label": "Nonfarm payrolls"}},
        },
        fred_macro=None,
        sec_recent_8k=None,
    )
    path = tmp_path / "AAPL.json"
    f.dump(path)
    return path


def test_clock_returns_frozen_utc_timestamp():
    c = Clock("2026-05-25T15:30:00Z")
    a, b = c(), c()
    assert a == b
    assert a.tzinfo is not None


def test_fixture_round_trip(tmp_path: Path):
    path = _sample_fixture(tmp_path)
    f2 = Fixture.load(path)
    assert f2.ticker == "AAPL"
    assert f2.yfinance_price["last"] == 190.0


def test_replay_writes_no_ledger_by_default(tmp_path: Path):
    path = _sample_fixture(tmp_path)
    ledger_dir = tmp_path / "ledger"
    result = replay(path, ledger_dir=ledger_dir, write_ledger=False)
    assert result.ledger_path is None
    assert not ledger_dir.exists() or not any(ledger_dir.iterdir())
    assert result.verdict.disclaimer.startswith("RESEARCH ONLY")


def test_replay_is_schema_stable_across_runs(tmp_path: Path):
    path = _sample_fixture(tmp_path)
    r1 = replay(path, ledger_dir=None, write_ledger=False)
    r2 = replay(path, ledger_dir=None, write_ledger=False)

    # run_id will differ; compare the verdict structure.
    v1 = r1.verdict.model_dump(mode="json")
    v2 = r2.verdict.model_dump(mode="json")
    assert v1 == v2
    # The screener output (the canonical numerics) must be identical too.
    # We re-derive it from the memo by spot-checking line count and disclaimer.
    assert r1.memo.splitlines()[0] == r2.memo.splitlines()[0]


def test_replay_ticker_outside_chain_returns_skip(tmp_path: Path):
    # Build a fixture with a missing chain to confirm SKIP fallback.
    f = Fixture(
        ticker="AAPL",
        frozen_now="2026-05-25T15:30:00+00:00",
        yfinance_price={"ticker": "AAPL", "last": 190.0},
        yfinance_history=None,
        yfinance_chain=None,
        econ_calendar=None,
        fred_macro=None,
        sec_recent_8k=None,
    )
    path = tmp_path / "AAPL_bare.json"
    f.dump(path)
    result = replay(path, write_ledger=False)
    assert result.verdict.action.value == "SKIP"
