from __future__ import annotations

from datetime import datetime, timezone

import pytest

from optagent.adapters.news_factual_yahoo import (
    YahooNewsAdapter,
    excerpts_from_envelope,
)
from optagent.profiles import ensure_default_profiles
from optagent.registry import ProviderRegistry
from optagent.schemas import Confidence, RunConfig, RunMode


def _registry(run_mode=RunMode.personal_research) -> ProviderRegistry:
    r = ProviderRegistry()
    ensure_default_profiles(r)
    r.bind(RunConfig(ticker="AAPL", run_mode=run_mode))
    return r


def _fake_news_payload(n: int = 2) -> list[dict]:
    return [
        {
            "id": f"id-{i}",
            "content": {
                "id": f"id-{i}",
                "contentType": "STORY",
                "title": f"Test headline {i}",
                "summary": f"Test summary {i} with some context.",
                "pubDate": "2026-05-25T06:00:00Z",
                "displayTime": "2026-05-25T06:00:00Z",
                "provider": {"displayName": f"Publisher {i}"},
                "clickThroughUrl": {"url": f"https://example.com/{i}"},
            },
        }
        for i in range(n)
    ]


class _FakeYF:
    def __init__(self, news: list[dict]) -> None:
        self.news = news

    def Ticker(self, ticker: str):  # noqa: N802 (yfinance API)
        class _T:
            news = self.news

        return _T


def test_news_adapter_returns_envelope_with_items():
    yf = _FakeYF(_fake_news_payload(3))
    adapter = YahooNewsAdapter(_registry(), yf_module=yf)
    env = adapter.get_news("AAPL", limit=5)
    assert env.confidence is Confidence.ok
    items = env.value["items"]
    assert len(items) == 3
    assert items[0]["title"] == "Test headline 0"
    assert items[0]["publisher"] == "Publisher 0"
    assert items[0]["url"].startswith("https://")


def test_news_adapter_truncates_long_text():
    long_title = "x" * 1000
    payload = [{"id": "z", "content": {"title": long_title, "summary": "y" * 2000, "provider": {"displayName": "P"}}}]
    adapter = YahooNewsAdapter(_registry(), yf_module=_FakeYF(payload))
    env = adapter.get_news("AAPL")
    item = env.value["items"][0]
    assert len(item["title"]) <= 280
    assert len(item["summary"]) <= 600


def test_news_adapter_empty_payload_marks_degraded():
    adapter = YahooNewsAdapter(_registry(), yf_module=_FakeYF([]))
    env = adapter.get_news("AAPL")
    assert env.confidence is Confidence.degraded
    assert env.value["items"] == []


def test_news_adapter_yfinance_unavailable_marks_unavailable():
    adapter = YahooNewsAdapter(_registry(), yf_module=None)
    # Constructor falls through to real import; if yfinance is installed
    # in the test env we cannot easily trigger the unavailable path here.
    # Force it by clearing the cached module reference:
    adapter._yf = None
    env = adapter.get_news("AAPL")
    assert env.confidence is Confidence.unavailable


def test_news_adapter_distributed_mode_blocks_research_only():
    yf = _FakeYF(_fake_news_payload())
    adapter = YahooNewsAdapter(_registry(run_mode=RunMode.distributed), yf_module=yf)
    env = adapter.get_news("AAPL")
    assert env.confidence is Confidence.unavailable


def test_excerpts_carry_unique_ids_per_item():
    yf = _FakeYF(_fake_news_payload(3))
    adapter = YahooNewsAdapter(_registry(), yf_module=yf)
    env = adapter.get_news("AAPL", limit=3)
    excerpts = excerpts_from_envelope(env)
    assert len(excerpts) == 3
    ids = [tc for tc, _ in excerpts]
    assert len(set(ids)) == 3
    # Each id pins back to the envelope's tool_call_id.
    assert all(tc.startswith(env.tool_call_id) for tc in ids)


def test_excerpts_empty_when_envelope_unavailable():
    yf = _FakeYF([])
    adapter = YahooNewsAdapter(_registry(), yf_module=yf)
    env = adapter.get_news("AAPL")  # empty -> degraded with items=[]
    assert excerpts_from_envelope(env) == []
