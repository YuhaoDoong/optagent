"""volume_oi_context — supply/demand proxy from the options chain.

DERIVED, NOT FETCHED. This adapter consumes a chain envelope produced by
another adapter (yfinance / moomoo) and computes:
  - Max Pain (the strike that minimises total intrinsic value at expiry)
  - Call wall   (strike with the largest call open interest)
  - Put wall    (strike with the largest put open interest)
  - PCR         (put-call ratio of OI)
  - PCR_volume  (put-call ratio of one-day volume)
  - OI center-of-mass distance from spot

The output Envelope carries a mandatory caveat — these are POSITIONING
PROXIES, NOT holder cost-basis ("chip distribution"). The validator's
(h) presence check enforces that the caveat appears in the rendered memo
when this envelope is cited.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Mapping

from ..registry import ProviderRegistry
from ..schemas import Confidence, Envelope, MarketSession


VOLUME_OI_CONTEXT_PROFILE_ID = "volume_oi_context_derived"


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


def _max_pain(rows: list[Mapping]) -> tuple[float | None, float | None]:
    """Return (max_pain_strike, total_intrinsic_at_that_strike)."""

    by_strike: dict[float, dict[str, int]] = {}
    for r in rows:
        strike = float(r.get("strike", 0) or 0)
        oi = int(r.get("open_interest", 0) or 0)
        if strike <= 0 or oi <= 0:
            continue
        right = r.get("right")
        slot = by_strike.setdefault(strike, {"call_oi": 0, "put_oi": 0})
        if right == "call":
            slot["call_oi"] += oi
        elif right == "put":
            slot["put_oi"] += oi

    if not by_strike:
        return None, None

    strikes = sorted(by_strike.keys())
    best_strike = None
    best_pain = None
    for s in strikes:
        # Total option holder payoff at expiration S
        # call holder payoff: max(S - K, 0) × oi   (writer pain)
        # put holder payoff:  max(K - S, 0) × oi
        pain = 0.0
        for k, slot in by_strike.items():
            if k < s:
                pain += (s - k) * slot["call_oi"]
            elif k > s:
                pain += (k - s) * slot["put_oi"]
        if best_pain is None or pain < best_pain:
            best_pain = pain
            best_strike = s
    return best_strike, best_pain


def _wall(rows: list[Mapping], right: str) -> tuple[float | None, int]:
    by_strike: dict[float, int] = {}
    for r in rows:
        if r.get("right") != right:
            continue
        strike = float(r.get("strike", 0) or 0)
        oi = int(r.get("open_interest", 0) or 0)
        if strike <= 0 or oi <= 0:
            continue
        by_strike[strike] = by_strike.get(strike, 0) + oi
    if not by_strike:
        return None, 0
    strike = max(by_strike, key=lambda k: by_strike[k])
    return strike, by_strike[strike]


def _pcr(rows: list[Mapping], field: str) -> float | None:
    call_total = 0
    put_total = 0
    for r in rows:
        amt = int(r.get(field, 0) or 0)
        if r.get("right") == "call":
            call_total += amt
        elif r.get("right") == "put":
            put_total += amt
    if call_total <= 0:
        return None
    return put_total / call_total


def _oi_center_of_mass(rows: list[Mapping]) -> float | None:
    total_oi = 0.0
    weighted = 0.0
    for r in rows:
        oi = int(r.get("open_interest", 0) or 0)
        strike = float(r.get("strike", 0) or 0)
        if oi <= 0 or strike <= 0:
            continue
        total_oi += oi
        weighted += oi * strike
    if total_oi <= 0:
        return None
    return weighted / total_oi


class VolumeOIContextAdapter:
    profile_id = VOLUME_OI_CONTEXT_PROFILE_ID

    CAVEAT = (
        "volume_oi_context is a derived options-positioning proxy "
        "(Max Pain / OI walls / PCR). It is NOT a holder cost-basis "
        "('chip distribution') measure."
    )

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._registry = registry
        self._now = now

    def _envelope(self, value, confidence: Confidence, warnings: list[str] | None = None) -> Envelope:
        now = self._now()
        return Envelope(
            value=value,
            as_of=now,
            source="volume_oi_context",
            delay_assumption="derived_from_chain",
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

    def compute(self, chain_value: Mapping | None, *, spot: float | None) -> Envelope:
        """Derive positioning context from the chain envelope's `.value`.

        Pass the chain envelope's `value` directly (it has the `rows` list).
        """

        blocked = self._check_gate()
        if blocked is not None:
            return blocked

        if not chain_value or not chain_value.get("rows"):
            return self._envelope(None, Confidence.unavailable, ["no_chain_rows"])

        rows = chain_value["rows"]
        max_pain_strike, max_pain_total = _max_pain(rows)
        call_wall_strike, call_wall_oi = _wall(rows, "call")
        put_wall_strike, put_wall_oi = _wall(rows, "put")
        pcr_oi = _pcr(rows, "open_interest")
        pcr_volume = _pcr(rows, "volume")
        com = _oi_center_of_mass(rows)

        com_distance_pct: float | None = None
        if com is not None and spot is not None and spot > 0:
            com_distance_pct = (com - spot) / spot

        if max_pain_strike is None and call_wall_strike is None and put_wall_strike is None:
            return self._envelope(None, Confidence.unavailable, ["no_positioning_signal"])

        value = {
            "caveat": self.CAVEAT,
            "expiration": chain_value.get("expiration"),
            "max_pain_strike": max_pain_strike,
            "max_pain_total_intrinsic": max_pain_total,
            "call_wall": {"strike": call_wall_strike, "oi": call_wall_oi},
            "put_wall": {"strike": put_wall_strike, "oi": put_wall_oi},
            "pcr_oi": pcr_oi,
            "pcr_volume": pcr_volume,
            "oi_center_of_mass": com,
            "oi_center_distance_pct_from_spot": com_distance_pct,
            "spot_used": spot,
        }
        return self._envelope(value, Confidence.ok)
