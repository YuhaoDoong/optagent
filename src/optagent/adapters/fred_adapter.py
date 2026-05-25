"""FRED macro adapter.

Fetches a small set of FRED series via the `fredapi` package. Every output is
wrapped in an Envelope tagged with `fred_default` (production_safe with
mandatory attribution). The adapter:
  - is gated by `ProviderRegistry.gate(...)` at every call;
  - returns `confidence=unavailable` envelopes instead of raising;
  - exposes a class-level `SERIES_IDS` map so callers can read which series
    are in scope without instantiating.

The user must supply a FRED API key via env var (`FRED_API_KEY`) or the
`api_key` constructor argument. When the key is missing the adapter degrades
to unavailable rather than failing the run.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Callable

from ..registry import ProviderRegistry
from ..schemas import Confidence, Envelope, MarketSession


FRED_PROFILE_ID = "fred_default"


# Curated series used in the v1 macro envelope. Keep this small so the
# audit ledger stays compact; the LLM only needs directional context.
SERIES_IDS: dict[str, str] = {
    "ten_year_yield": "DGS10",
    "two_year_yield": "DGS2",
    "vix_close": "VIXCLS",
    "cpi_yoy": "CPIAUCSL",
    "fed_funds": "FEDFUNDS",
    "usd_index": "DTWEXBGS",
}


# Per-series source citation (FRED ToS expects each upstream source to be
# named when republishing). Keys are FRED series ids (the values of
# `SERIES_IDS`, NOT the friendly keys).
SERIES_SOURCES: dict[str, str] = {
    "DGS10": "Board of Governors of the Federal Reserve System (US)",
    "DGS2": "Board of Governors of the Federal Reserve System (US)",
    "VIXCLS": "Chicago Board Options Exchange (CBOE)",
    "CPIAUCSL": "U.S. Bureau of Labor Statistics",
    "FEDFUNDS": "Board of Governors of the Federal Reserve System (US)",
    "DTWEXBGS": "Board of Governors of the Federal Reserve System (US)",
}


class FREDUnavailableError(RuntimeError):
    """Raised at adapter init when fredapi cannot be imported AND no client is injected."""


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


class FREDAdapter:
    profile_id = FRED_PROFILE_ID

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        fred_client: Any | None = None,
        api_key: str | None = None,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._registry = registry
        self._now = now
        if fred_client is not None:
            self._fred = fred_client
            self._api_key = "<injected>"
            return
        key = api_key or os.environ.get("FRED_API_KEY")
        self._api_key = key
        if not key:
            self._fred = None
            return
        try:
            from fredapi import Fred  # noqa: WPS433
        except ImportError as e:
            raise FREDUnavailableError(
                "fredapi is not installed; install with `pip install fredapi` or pass fred_client=..."
            ) from e
        self._fred = Fred(api_key=key)

    # ------------------------------------------------------------------
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
            source="fred",
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
    def get_macro(self) -> Envelope:
        """Fetch the curated FRED series. Ticker-agnostic (cached once per run)."""

        blocked = self._check_gate()
        if blocked is not None:
            return blocked
        if self._fred is None:
            return self._envelope(None, Confidence.unavailable, ["no_FRED_API_KEY"])

        readings: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        for name, series_id in SERIES_IDS.items():
            try:
                series = self._fred.get_series_latest_release(series_id)
                # pandas Series — pluck the last value and its index timestamp.
                if series is None or len(series) == 0:
                    warnings.append(f"{series_id}_empty")
                    continue
                latest_value = float(series.iloc[-1])
                latest_ts = series.index[-1]
                # latest_ts is a pandas Timestamp; render as ISO 8601 UTC date.
                ts_iso = (
                    latest_ts.tz_localize("UTC").isoformat()
                    if getattr(latest_ts, "tz", None) is None
                    else latest_ts.tz_convert("UTC").isoformat()
                )
                readings[name] = {
                    "series_id": series_id,
                    "value": latest_value,
                    "observation_date": ts_iso,
                    "source": SERIES_SOURCES.get(series_id, "unknown"),
                }
            except Exception as e:  # noqa: BLE001 - degrade, don't raise
                warnings.append(f"{series_id}_fetch_failed:{e.__class__.__name__}")

        if not readings:
            return self._envelope(None, Confidence.unavailable, warnings or ["all_series_failed"])

        # Aggregate per-series source attributions so the renderer / validator
        # can surface them when the envelope is cited. Each entry is the
        # canonical "<source>, via FRED" attribution.
        series_attributions = sorted({
            f"{r['source']}, retrieved from FRED (series {r['series_id']})"
            for r in readings.values()
        })

        confidence = Confidence.degraded if warnings else Confidence.ok
        return self._envelope(
            {"readings": readings, "series_attributions": series_attributions},
            confidence,
            warnings,
        )
