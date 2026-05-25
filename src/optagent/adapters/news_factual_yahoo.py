"""News-factual adapter backed by yfinance's free Yahoo News feed.

`yfinance.Ticker(...).news` returns a list of headline metadata objects
under Yahoo Finance's research/personal-use terms. The adapter extracts
**factual snippets only** — title, summary, publisher name, publication
timestamp — and never asks the LLM to summarise or sentiment-score the
text. AC-8's prompt-injection defence wraps each excerpt in
`<news_excerpt id="...">...</news_excerpt>` delimiters before reaching
the LLM (see `optagent.llm.build_user_prompt`).

The adapter inherits the existing `yfinance_research` provider profile
because the data comes through the same Yahoo channel and is subject to
the same personal-use restriction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from ..registry import ProviderRegistry
from ..schemas import Confidence, Envelope, MarketSession


NEWS_PROFILE_ID = "yfinance_news_research"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _classify_session(now: datetime) -> MarketSession:
    if now.weekday() >= 5:
        return MarketSession.closed
    hour = now.hour
    if 13 <= hour <= 20:
        return MarketSession.rth
    if 11 <= hour < 13:
        return MarketSession.pre_market
    if 20 < hour <= 23:
        return MarketSession.after_hours
    return MarketSession.closed


class YahooNewsAdapter:
    """News headlines via the yfinance package's Yahoo News API.

    The adapter is intentionally minimal: it does NOT classify sentiment,
    summarise content, or rank stories — those would expose the LLM to
    prompt-injection vectors. Callers get a flat list of headline metadata
    for downstream prompt-builder wrapping.
    """

    profile_id = NEWS_PROFILE_ID

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        yf_module: Any | None = None,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._registry = registry
        self._now = now
        if yf_module is not None:
            self._yf = yf_module
            return
        try:
            import yfinance as yf  # noqa: WPS433
        except ImportError:
            self._yf = None
            return
        self._yf = yf

    def _envelope(
        self, value: Any | None, confidence: Confidence, warnings: list[str] | None = None
    ) -> Envelope:
        now = self._now()
        return Envelope(
            value=value,
            as_of=now,
            source="yahoo_news",
            delay_assumption="best_effort",
            market_session=_classify_session(now),
            confidence=confidence,
            provider_profile_id=self.profile_id,
            warnings=warnings or [],
        )

    def _check_gate(self) -> Envelope | None:
        gate = self._registry.gate(self.profile_id)
        if not gate.ok:
            return self._envelope(None, Confidence.unavailable, [f"compliance_gate_blocked: {gate.reason}"])
        return None

    def get_news(self, ticker: str, limit: int = 5) -> Envelope:
        """Return the most-recent `limit` headlines for `ticker`.

        Each item carries the minimum factual surface: id / title / summary
        / publisher / pub_date / url. Body text is never scraped or fetched.
        """

        blocked = self._check_gate()
        if blocked is not None:
            return blocked
        if self._yf is None:
            return self._envelope(None, Confidence.unavailable, ["yfinance_not_installed"])

        try:
            tk = self._yf.Ticker(ticker)
            raw = tk.news or []
        except Exception as e:  # noqa: BLE001 - degrade, don't raise
            return self._envelope(None, Confidence.unavailable, [f"news_fetch_failed:{e.__class__.__name__}"])

        items: list[dict[str, Any]] = []
        for entry in raw[:limit]:
            content = entry.get("content") if isinstance(entry, dict) else None
            if not isinstance(content, dict):
                continue
            provider = content.get("provider") or {}
            publisher = provider.get("displayName") if isinstance(provider, dict) else None
            click_through = content.get("clickThroughUrl") or {}
            canonical = content.get("canonicalUrl") or {}
            url = (
                (click_through.get("url") if isinstance(click_through, dict) else None)
                or (canonical.get("url") if isinstance(canonical, dict) else None)
            )
            items.append(
                {
                    "id": str(entry.get("id") or content.get("id") or ""),
                    "title": str(content.get("title") or "")[:280],
                    "summary": str(content.get("summary") or "")[:600],
                    "publisher": str(publisher or "unknown"),
                    "pub_date": str(content.get("pubDate") or content.get("displayTime") or ""),
                    "url": str(url or ""),
                }
            )

        if not items:
            return self._envelope(
                {"ticker": ticker, "items": []},
                Confidence.degraded,
                ["no_news"],
            )
        return self._envelope({"ticker": ticker, "items": items}, Confidence.ok)


def _strip_delimiters(text: str) -> str:
    """Close the delimiter-escape vector Codex R3 flagged.

    If an article body contained the literal `</news_excerpt>` (or its open
    variant), the LLM could read it as the end of the data section and
    interpret following text as instructions. We replace the angle brackets
    with safe inline placeholders before wrapping.
    """

    return (
        text
        .replace("<news_excerpt", "&lt;news_excerpt")
        .replace("</news_excerpt", "&lt;/news_excerpt")
        .replace("<sec_excerpt", "&lt;sec_excerpt")
        .replace("</sec_excerpt", "&lt;/sec_excerpt")
    )


def excerpts_from_envelope(env: Envelope, max_chars: int = 200) -> list[tuple[str, str]]:
    """Turn a news envelope into `(tool_call_id, excerpt_text)` tuples.

    Used by the orchestrator to feed the LLM prompt-builder. The excerpt
    is the title + a truncated summary — short, factual, and bounded so
    a single excessively-long article cannot consume the LLM's budget.

    The returned ids carry an `#item-N` suffix so an audit consumer can map
    the LLM's cited string back to a specific headline. The validator
    correctly rejects them as `tool_call_id`s (they're not envelope ids);
    LLMs should cite the parent envelope's id instead.
    """

    if env.value is None:
        return []
    items = env.value.get("items") or []
    if not items:
        return []
    out: list[tuple[str, str]] = []
    for i, item in enumerate(items):
        title = _strip_delimiters(str(item.get("title", "")))
        body_lines = [f"title: {title}"]
        if item.get("summary"):
            summary = _strip_delimiters(str(item["summary"]))[:max_chars]
            body_lines.append(f"summary: {summary}")
        if item.get("publisher"):
            body_lines.append(f"publisher: {item['publisher']}")
        if item.get("pub_date"):
            body_lines.append(f"published: {item['pub_date']}")
        excerpt = "\n".join(body_lines)
        out.append((f"{env.tool_call_id}#item-{i}", excerpt))
    return out
