"""Regression tests for the Codex web-UI audit findings (Round 13)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest


def _fake_yf_module(expiry_iso: str, calls: list[dict], puts: list[dict], last: float = 190.0):
    """Minimal yfinance shim — same shape used elsewhere in the suite."""

    class _FastInfo:
        last_price = last

        def __getitem__(self, k):
            return getattr(self, k)

        def get(self, k, default=None):
            return getattr(self, k, default)

    class _FakeDF:
        def __init__(self, rows):
            self._rows = rows

        def itertuples(self, index=False):
            from types import SimpleNamespace

            for r in self._rows:
                yield SimpleNamespace(**r)

    class _Chain:
        def __init__(self, calls, puts):
            self.calls = _FakeDF(calls)
            self.puts = _FakeDF(puts)

    class _FakeTicker:
        def __init__(self, *_a, **_kw):
            self.options = (expiry_iso,)
            self.fast_info = _FastInfo()

        def option_chain(self, exp):
            return _Chain(calls, puts)

        def history(self, **_kw):
            closes = [180 + i for i in range(30)]
            idx = pd.date_range("2026-03-01", periods=30, freq="B")
            return pd.DataFrame({"Close": closes}, index=idx)

    class _M:
        Ticker = _FakeTicker

    return _M()


def _row(symbol, strike, bid, ask, oi=5000, volume=300, iv=0.28):
    return {
        "contractSymbol": symbol,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "lastPrice": (bid + ask) / 2.0,
        "volume": volume,
        "openInterest": oi,
        "impliedVolatility": iv,
    }


def test_analyze_result_carries_envelopes_directly(tmp_path: Path):
    """Codex web-audit: AnalyzeResult must expose envelopes / screener
    candidates / ml_signal so the UI doesn't have to re-read the shared
    ledger (race condition vector)."""

    from optagent.adapters import YFinanceAdapter
    from optagent.orchestrator import analyze
    from optagent.profiles import ensure_default_profiles
    from optagent.registry import ProviderRegistry

    expiry = (datetime.now(timezone.utc).date() + timedelta(days=21)).isoformat()
    yf = _fake_yf_module(
        expiry,
        calls=[_row("AAPL_C200", 200, 2.40, 2.60), _row("AAPL_C210", 210, 1.40, 1.50)],
        puts=[_row("AAPL_P180", 180, 1.20, 1.40)],
    )
    registry = ProviderRegistry()
    ensure_default_profiles(registry)
    adapter = YFinanceAdapter(registry, yf_module=yf)
    result = analyze(
        "AAPL",
        registry=registry,
        yfinance_adapter=adapter,
        ledger_dir=tmp_path,
    )

    # Codex fix: result must carry envelopes IN-PROCESS — no ledger re-read.
    assert isinstance(result.envelopes, list)
    assert len(result.envelopes) >= 2  # at least price + chain
    assert all(e.source for e in result.envelopes)

    # Screener candidates also surfaced directly.
    assert isinstance(result.screener_candidates, list)

    # ml_signal is None because we didn't pass an ML adapter.
    assert result.ml_signal is None


def test_web_does_not_write_credentials_to_environ():
    """Codex web-audit: the UI must NOT mutate os.environ with sidebar
    secrets. This test grep-checks the app.py source as a fast guard."""

    from pathlib import Path

    app_path = Path(__file__).resolve().parents[1] / "src" / "optagent" / "web" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    # The OLD code had `os.environ["OPTAGENT_USER_AGENT"] = ...`
    assert 'os.environ["OPTAGENT_USER_AGENT"]' not in source
    assert 'os.environ["FRED_API_KEY"]' not in source
    # New code passes credentials directly to adapter constructors.
    assert "user_agent=sidebar_opts" in source
    assert "api_key=sidebar_opts" in source


def test_web_does_not_reopen_ledger_for_envelopes():
    """Codex web-audit: the UI must read result.envelopes, NOT re-open the
    shared ledger file (race condition with concurrent users)."""

    from pathlib import Path

    app_path = Path(__file__).resolve().parents[1] / "src" / "optagent" / "web" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    # The OLD code had `result.ledger_path.open("r"...` + `readlines()[-1]`
    assert "readlines()[-1]" not in source
    # The new code uses the in-process AnalyzeResult fields.
    assert "result.envelopes" in source
    assert "result.screener_candidates" in source


def test_verdict_card_hex_color_validation_in_source():
    """The badge HTML sink must defensively validate the color."""

    from pathlib import Path

    app_path = Path(__file__).resolve().parents[1] / "src" / "optagent" / "web" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "html.escape" in source or "_html.escape" in source
    assert "#RRGGBB" in source or '"0123456789abcdefABCDEF"' in source
