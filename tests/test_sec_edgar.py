from __future__ import annotations

import pytest

from optagent.adapters.sec_edgar_adapter import SECEdgarAdapter
from optagent.profiles import ensure_default_profiles
from optagent.registry import ProviderRegistry
from optagent.schemas import Confidence, RunConfig, RunMode


def _registry() -> ProviderRegistry:
    r = ProviderRegistry()
    ensure_default_profiles(r)
    r.bind(RunConfig(ticker="AAPL"))
    return r


TICKER_MAP_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
}

AAPL_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "8-K", "8-K", "10-Q", "8-K"],
            "filingDate": ["2026-04-01", "2026-04-15", "2026-04-20", "2026-04-25", "2026-05-01"],
            "accessionNumber": ["a1", "a2", "a3", "a4", "a5"],
            "primaryDocument": ["d1.htm", "d2.htm", "d3.htm", "d4.htm", "d5.htm"],
        }
    }
}


def _make_adapter(captured_calls: list[tuple[str, dict]]):
    def fake_http_get(url: str, headers: dict[str, str]):
        captured_calls.append((url, headers))
        if "company_tickers" in url:
            return TICKER_MAP_PAYLOAD
        if "submissions/CIK" in url:
            return AAPL_SUBMISSIONS
        return None

    return SECEdgarAdapter(
        _registry(),
        http_get=fake_http_get,
        user_agent="optagent-test/0.0.1 (test@example.com)",
    )


def test_recent_8k_returns_only_8k_filings():
    calls: list = []
    adapter = _make_adapter(calls)
    env = adapter.get_recent_8k("AAPL", limit=10)
    assert env.confidence is Confidence.ok
    assert env.value["ticker"] == "AAPL"
    assert env.value["cik"] == "0000320193"
    forms = [f["form"] for f in env.value["recent_8k"]]
    assert forms == ["8-K", "8-K", "8-K"]


def test_user_agent_header_is_present_on_every_call():
    calls: list = []
    adapter = _make_adapter(calls)
    adapter.get_recent_8k("AAPL")
    assert len(calls) == 2
    for _, headers in calls:
        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("optagent-test")


def test_unknown_ticker_marks_unavailable():
    calls: list = []
    adapter = _make_adapter(calls)
    env = adapter.get_recent_8k("ZZZZ")
    assert env.confidence is Confidence.unavailable
    assert "no_cik" in (env.warnings[0] if env.warnings else "")


def test_http_failure_marks_unavailable():
    def boom(url, headers):
        raise RuntimeError("network")

    adapter = SECEdgarAdapter(
        _registry(),
        http_get=boom,
        user_agent="optagent-test/0.0.1 (test@example.com)",
    )
    env = adapter.get_recent_8k("AAPL")
    assert env.confidence is Confidence.unavailable


def test_limit_caps_returned_filings():
    calls: list = []
    adapter = _make_adapter(calls)
    env = adapter.get_recent_8k("AAPL", limit=2)
    assert len(env.value["recent_8k"]) == 2
