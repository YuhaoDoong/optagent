"""SEC EDGAR adapter.

Fetches recent 8-K filings (and the company's CIK) for a given ticker using
SEC EDGAR's public JSON endpoints. SEC's developer guidance REQUIRES a
descriptive `User-Agent` header that identifies the requester and a polite
rate limit (≤10 req/s per host). The adapter enforces both.

EDGAR endpoints used:
  - https://www.sec.gov/files/company_tickers.json
        ticker -> CIK lookup (all tickers in one ~2 MB file)
  - https://data.sec.gov/submissions/CIK{padded}.json
        recent submissions for one CIK

We do NOT scrape filing bodies in v1 — only metadata (form, date, accession).
The LLM gets a short factual summary; no sentiment scoring per AC-8.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from ..registry import ProviderRegistry
from ..schemas import Confidence, Envelope, MarketSession


SEC_PROFILE_ID = "sec_edgar_default"
DEFAULT_USER_AGENT_HINT = (
    "optagent-research/0.0.1 (set OPTAGENT_USER_AGENT env var to your contact email)"
)
MIN_INTERVAL_S = 0.11  # 600 / 60 = 10 rps ceiling; we go slightly under


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


class _RateLimiter:
    """Thread-unsafe but adequate for single-process CLI use."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last_call_at = 0.0

    def wait(self, now_monotonic: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        delta = now_monotonic() - self._last_call_at
        if delta < self.min_interval_s:
            sleep(self.min_interval_s - delta)
        self._last_call_at = now_monotonic()


class SECEdgarAdapter:
    profile_id = SEC_PROFILE_ID

    BASE_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
    BASE_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        http_get: Callable[[str, dict[str, str]], dict[str, Any] | None] | None = None,
        user_agent: str | None = None,
        now: Callable[[], datetime] = _utc_now,
        rate_limiter: _RateLimiter | None = None,
    ) -> None:
        import os

        self._registry = registry
        self._now = now
        self._user_agent = (
            user_agent
            or os.environ.get("OPTAGENT_USER_AGENT")
            or DEFAULT_USER_AGENT_HINT
        )
        self._rate_limiter = rate_limiter or _RateLimiter(MIN_INTERVAL_S)
        self._http_get = http_get or self._default_http_get
        self._ticker_cache: dict[str, str] | None = None

    # ------------------------------------------------------------------
    def _default_http_get(self, url: str, headers: dict[str, str]) -> dict[str, Any] | None:
        import gzip
        import json
        import urllib.request
        import zlib

        req = urllib.request.Request(url, headers=headers)
        self._rate_limiter.wait()
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            encoding = resp.headers.get("Content-Encoding", "").lower()
        if encoding == "gzip":
            data = gzip.decompress(data)
        elif encoding == "deflate":
            data = zlib.decompress(data)
        return json.loads(data)

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": None,  # filled by stdlib; placeholder for visibility
        }

    def _envelope(
        self,
        value: Any | None,
        confidence: Confidence,
        warnings: list[str] | None = None,
    ) -> Envelope:
        now = self._now()
        return Envelope(
            value=value,
            as_of=now,
            source="sec_edgar",
            delay_assumption="eod",
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

    # ------------------------------------------------------------------
    def _load_ticker_map(self) -> dict[str, str] | None:
        """Return ticker -> zero-padded CIK string. None on fetch failure."""

        if self._ticker_cache is not None:
            return self._ticker_cache
        try:
            headers = {k: v for k, v in self._headers().items() if v is not None}
            payload = self._http_get(self.BASE_TICKER_URL, headers)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        out: dict[str, str] = {}
        for entry in payload.values():
            try:
                ticker = str(entry["ticker"]).upper()
                cik = int(entry["cik_str"])
            except (KeyError, TypeError, ValueError):
                continue
            out[ticker] = f"{cik:010d}"
        self._ticker_cache = out
        return out

    # ------------------------------------------------------------------
    def get_recent_8k(self, ticker: str, limit: int = 5) -> Envelope:
        """Return the last ~`limit` 8-K filings for `ticker` (metadata only)."""

        blocked = self._check_gate()
        if blocked is not None:
            return blocked

        ticker = ticker.upper()
        ticker_map = self._load_ticker_map()
        if not ticker_map:
            return self._envelope(None, Confidence.unavailable, ["ticker_map_unavailable"])

        cik = ticker_map.get(ticker)
        if not cik:
            return self._envelope(None, Confidence.unavailable, [f"no_cik_for_{ticker}"])

        url = self.BASE_SUBMISSIONS_URL.format(cik=cik)
        try:
            headers = {k: v for k, v in self._headers().items() if v is not None}
            payload = self._http_get(url, headers)
        except Exception as e:  # noqa: BLE001
            return self._envelope(None, Confidence.unavailable, [f"submissions_fetch_failed:{e.__class__.__name__}"])

        if not isinstance(payload, dict):
            return self._envelope(None, Confidence.unavailable, ["submissions_bad_shape"])

        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        filings: list[dict[str, str]] = []
        for form, dt, acc, doc in zip(forms, dates, accessions, primary_docs):
            if form != "8-K":
                continue
            filings.append(
                {
                    "form": form,
                    "filing_date": dt,
                    "accession": acc,
                    "primary_document": doc,
                }
            )
            if len(filings) >= limit:
                break

        if not filings:
            return self._envelope(
                {"ticker": ticker, "cik": cik, "recent_8k": []},
                Confidence.degraded,
                ["no_recent_8k"],
            )
        return self._envelope(
            {"ticker": ticker, "cik": cik, "recent_8k": filings},
            Confidence.ok,
        )
