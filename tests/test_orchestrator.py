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


_TTL_TABLE = {
    "price": {"rth": 10, "after_hours": 300, "critical": True},
    "options_chain": {
        "rth_low_vol": 30,
        "rth_high_vol_or_near_expiry": 15,
        "after_hours": 300,
        "critical": True,
    },
}
_PRICE_TABLE = {
    "price_table_version": "test-1",
    "default_model": "claude-haiku-4-5",
    "limits": {
        "max_input_tokens": 60000,
        "max_output_tokens": 2000,
        "max_retries": 2,
        "timeout_s": 45,
        "safety_margin": 0.20,
        "cap_usd": 5.00,
    },
    "models": {
        "claude-haiku-4-5": {
            "input_usd_per_mtok": 0.80,
            "output_usd_per_mtok": 4.0,
            "tokenizer_version": "claude-2026-04",
        },
    },
}


class _FakeLLMClient:
    """Returns the canned `tool_input` regardless of prompt."""

    def __init__(self, tool_input: dict) -> None:
        self._tool_input = tool_input
        self.calls = 0

    def synthesise(self, *, system, user_prompt, tool, max_output_tokens, timeout_s):
        self.calls += 1
        return self._tool_input, {"stop_reason": "tool_use"}


def _setup_liquid_run():
    expiry = _expiry_iso(20)
    calls = [
        _row("AAPL_C200", 200, 2.40, 2.60, oi=5000, volume=300),
        _row("AAPL_C210", 210, 1.40, 1.50, oi=4000, volume=200),
    ]
    puts = [_row("AAPL_P180", 180, 1.20, 1.40, oi=3000, volume=200)]
    yf = _fake_yf_module(expiry, calls, puts)
    registry = ProviderRegistry()
    adapter = YFinanceAdapter(registry, yf_module=yf)
    return registry, adapter


def test_llm_path_clean_long_call_passes_validator(tmp_path: Path):
    registry, adapter = _setup_liquid_run()
    client = _FakeLLMClient(
        {
            "direction": "LONG_CALL",
            "chosen_occ": "AAPL_C200",
            "conviction": 0.6,
            "primary_reasons": ["liquid ATM call within DTE window"],
            "tool_call_ids_used": [],  # filled in below from envelopes
        }
    )

    # Tap the adapter's tool_call_ids by calling once to seed citations.
    # Easier: use a fake that introspects later. We can't inject tcids here
    # without re-running, so let the client cite both envelopes by index.
    class _ClientWithDynamicTCIDs:
        def synthesise(self, *, system, user_prompt, tool, max_output_tokens, timeout_s):
            # Extract the two available tool_call_ids from the prompt body.
            import re

            tcids = re.findall(r"(tc-[0-9a-f]+)", user_prompt)
            tcids = list(dict.fromkeys(tcids))[:2]
            return (
                {
                    "direction": "LONG_CALL",
                    "chosen_occ": "AAPL_C200",
                    "conviction": 0.6,
                    "primary_reasons": ["liquid ATM call within DTE window"],
                    "tool_call_ids_used": tcids,
                },
                {"stop_reason": "tool_use"},
            )

    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
        enable_llm=True,
        llm_client=_ClientWithDynamicTCIDs(),
        model_version="claude-haiku-4-5",
        price_table=_PRICE_TABLE,
        ttl_table=_TTL_TABLE,
    )
    assert result.verdict.action is VerdictAction.long_call
    assert result.verdict.contract is not None
    assert result.verdict.contract.occ_symbol == "AAPL_C200"


def test_llm_path_synthesis_exception_fails_closed_to_skip(tmp_path: Path):
    """A client that raises (truncated JSON, network error, SDK bug) must NOT
    crash analyze(); it fails closed to a structured SKIP."""

    from optagent.schemas import SkipReason

    registry, adapter = _setup_liquid_run()

    class _BoomClient:
        def synthesise(self, *, system, user_prompt, tool, max_output_tokens, timeout_s):
            raise RuntimeError("OpenRouter returned malformed tool-call JSON (truncated).")

    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
        enable_llm=True,
        llm_client=_BoomClient(),
        model_version="claude-haiku-4-5",
        price_table=_PRICE_TABLE,
        ttl_table=_TTL_TABLE,
    )
    assert result.verdict.action is VerdictAction.skip
    assert result.verdict.skip_reason is SkipReason.critical_provider_unavailable
    assert any("synthesis failed" in r.lower() for r in result.verdict.primary_reasons)


def test_llm_path_hallucinated_occ_is_downgraded_to_skip(tmp_path: Path):
    registry, adapter = _setup_liquid_run()
    client = _FakeLLMClient(
        {
            "direction": "LONG_CALL",
            "chosen_occ": "AAPL_C999_HALLUCINATED",
            "primary_reasons": ["I made this up"],
            "tool_call_ids_used": [],
        }
    )
    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
        enable_llm=True,
        llm_client=client,
        model_version="claude-haiku-4-5",
        price_table=_PRICE_TABLE,
        ttl_table=_TTL_TABLE,
    )
    assert result.verdict.action is VerdictAction.skip
    # The LLM-builder catches phantom OCC up front; the validator catches it
    # if it sneaks past. Either way: SKIP.


def test_llm_path_unknown_model_falls_back_to_skip(tmp_path: Path):
    registry, adapter = _setup_liquid_run()
    client = _FakeLLMClient(
        {
            "direction": "LONG_CALL",
            "chosen_occ": "AAPL_C200",
            "primary_reasons": ["..."],
            "tool_call_ids_used": [],
        }
    )
    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
        enable_llm=True,
        llm_client=client,
        model_version="gpt-5-not-in-price-table",
        price_table=_PRICE_TABLE,
        ttl_table=_TTL_TABLE,
    )
    assert result.verdict.action is VerdictAction.skip
    assert client.calls == 0  # never reached the LLM


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
