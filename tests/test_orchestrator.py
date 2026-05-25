"""End-to-end orchestrator tests with a fake yfinance module.

The orchestrator must never call the network in tests; we inject a fake
`yfinance` shim whose `Ticker` returns deterministic chain data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from optagent.adapters import YFinanceAdapter
from optagent.ledger import read_all
from optagent.orchestrator import analyze
from optagent.registry import ProviderRegistry
from optagent.schemas import VerdictAction


@dataclass
class _FastInfo:
    last_price: float

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class _OptionChain:
    def __init__(self, calls: list[dict], puts: list[dict]) -> None:
        self.calls = _FakeDF(calls)
        self.puts = _FakeDF(puts)


class _FakeDF:
    """Minimal DataFrame stand-in supporting .itertuples(index=False)."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def itertuples(self, index: bool = False):
        from types import SimpleNamespace

        for r in self._rows:
            yield SimpleNamespace(**r)


def _fake_yf_module(expiry_iso: str, calls: list[dict], puts: list[dict], last: float = 190.0):
    class _FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker
            self.options = (expiry_iso,)
            self.fast_info = _FastInfo(last_price=last)

        def option_chain(self, expiry: str) -> _OptionChain:
            assert expiry == expiry_iso
            return _OptionChain(calls, puts)

    class _Module:
        Ticker = _FakeTicker

    return _Module()


def _row(
    contract: str,
    strike: float,
    bid: float,
    ask: float,
    oi: int,
    volume: int,
    iv: float = 0.25,
) -> dict:
    return {
        "contractSymbol": contract,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "lastPrice": (bid + ask) / 2.0,
        "volume": volume,
        "openInterest": oi,
        "impliedVolatility": iv,
    }


def _expiry_iso(days: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()


def test_analyze_returns_skip_when_no_chain_rows(tmp_path: Path):
    expiry = _expiry_iso(20)
    yf = _fake_yf_module(expiry, calls=[], puts=[])
    registry = ProviderRegistry()
    adapter = YFinanceAdapter(registry, yf_module=yf)
    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
    )
    assert result.verdict.action is VerdictAction.skip
    # Either critical-provider-unavailable (chain empty marks unavailable) or
    # no-candidates-after-screen; both are correct SKIPs here.
    assert result.verdict.skip_reason is not None
    assert result.ledger_path is not None
    assert result.ledger_path.exists()


def test_analyze_with_liquid_chain_writes_ledger(tmp_path: Path):
    expiry = _expiry_iso(20)
    calls = [
        _row("AAPL_C200", 200, 2.40, 2.60, oi=5000, volume=300),
        _row("AAPL_C210", 210, 1.40, 1.50, oi=4000, volume=200),
    ]
    puts = [
        _row("AAPL_P180", 180, 1.20, 1.40, oi=3000, volume=200),
    ]
    yf = _fake_yf_module(expiry, calls, puts)
    registry = ProviderRegistry()
    adapter = YFinanceAdapter(registry, yf_module=yf)
    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
    )

    assert result.memo.startswith("RESEARCH ONLY")
    assert result.ledger_path is not None
    records = read_all(result.ledger_path)
    assert len(records) == 1
    rec = records[0]
    assert rec.ticker == "AAPL"
    # Template-only neutral bias → still SKIP for v1 release.
    assert rec.final_verdict.action is VerdictAction.skip
    assert rec.screener_output  # candidates survived the screener


def test_analyze_no_ledger_flag(tmp_path: Path):
    expiry = _expiry_iso(20)
    yf = _fake_yf_module(expiry, calls=[], puts=[])
    registry = ProviderRegistry()
    adapter = YFinanceAdapter(registry, yf_module=yf)
    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
        write_ledger=False,
    )
    assert result.ledger_path is None
    assert not list(tmp_path.glob("*.jsonl"))


def test_analyze_no_expiry_in_window_returns_skip(tmp_path: Path):
    # Expiry 200 days out → outside default screener band of ≤ 3×horizon=42.
    expiry = _expiry_iso(200)
    yf = _fake_yf_module(expiry, calls=[_row("X", 200, 1, 2, 1000, 100)], puts=[])
    registry = ProviderRegistry()
    adapter = YFinanceAdapter(registry, yf_module=yf)
    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
    )
    assert result.verdict.action is VerdictAction.skip
