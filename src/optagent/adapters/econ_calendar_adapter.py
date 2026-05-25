"""Curated US macro-event calendar.

No network call. The calendar holds FOMC meetings, scheduled CPI / NFP / PPI
release dates, and ad-hoc geopolitical-risk windows the user wants the
screener to avoid. The adapter computes `days_to_next_*` integers per event
class and surfaces the next event in the envelope value.

Dates are intentionally code-bundled rather than scraped to keep the system
ToS-safe and entirely offline. Update the table once per quarter when the
official Fed / BLS schedules publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable

from ..registry import ProviderRegistry
from ..schemas import Confidence, Envelope, MarketSession


ECON_PROFILE_ID = "econ_calendar_builtin"


@dataclass(frozen=True)
class _Event:
    """Single calendar entry."""

    iso_date: str  # YYYY-MM-DD
    kind: str  # FOMC | CPI | NFP | PPI | GDP | OTHER
    label: str

    @property
    def event_date(self) -> date:
        return date.fromisoformat(self.iso_date)


# Hand-maintained schedule. Replace quarterly from federalreserve.gov + bls.gov.
BUILT_IN_EVENTS: list[_Event] = [
    _Event("2026-06-17", "FOMC", "FOMC meeting + SEP"),
    _Event("2026-07-29", "FOMC", "FOMC meeting"),
    _Event("2026-09-16", "FOMC", "FOMC meeting + SEP"),
    _Event("2026-10-28", "FOMC", "FOMC meeting"),
    _Event("2026-12-09", "FOMC", "FOMC meeting + SEP"),
    _Event("2026-06-11", "CPI", "CPI release"),
    _Event("2026-07-15", "CPI", "CPI release"),
    _Event("2026-08-12", "CPI", "CPI release"),
    _Event("2026-09-10", "CPI", "CPI release"),
    _Event("2026-10-15", "CPI", "CPI release"),
    _Event("2026-06-06", "NFP", "Nonfarm payrolls"),
    _Event("2026-07-03", "NFP", "Nonfarm payrolls"),
    _Event("2026-08-01", "NFP", "Nonfarm payrolls"),
    _Event("2026-09-05", "NFP", "Nonfarm payrolls"),
    _Event("2026-10-03", "NFP", "Nonfarm payrolls"),
    _Event("2026-06-12", "PPI", "PPI release"),
    _Event("2026-07-16", "PPI", "PPI release"),
    _Event("2026-08-13", "PPI", "PPI release"),
    _Event("2026-09-11", "PPI", "PPI release"),
    _Event("2026-06-26", "GDP", "GDP Q1 third estimate"),
    _Event("2026-07-30", "GDP", "GDP Q2 advance"),
    _Event("2026-08-28", "GDP", "GDP Q2 second"),
]


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


class EconCalendarAdapter:
    profile_id = ECON_PROFILE_ID

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        events: list[_Event] | None = None,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._registry = registry
        self._events: list[_Event] = sorted(
            events if events is not None else BUILT_IN_EVENTS,
            key=lambda e: e.iso_date,
        )
        self._now = now

    def _envelope(self, value: dict | None, confidence: Confidence, warnings: list[str] | None = None) -> Envelope:
        now = self._now()
        return Envelope(
            value=value,
            as_of=now,
            source="econ_calendar_builtin",
            delay_assumption="static_schedule",
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

    def get_calendar(self) -> Envelope:
        blocked = self._check_gate()
        if blocked is not None:
            return blocked

        today = self._now().date()
        future = [e for e in self._events if e.event_date >= today]
        if not future:
            return self._envelope({"next_event": None, "days_to_next_event": None}, Confidence.degraded, ["no_future_events"])

        days_by_kind: dict[str, int] = {}
        next_by_kind: dict[str, dict[str, str]] = {}
        for e in future:
            if e.kind not in days_by_kind:
                days_by_kind[e.kind] = (e.event_date - today).days
                next_by_kind[e.kind] = {"date": e.iso_date, "label": e.label}

        next_event = future[0]
        days_to_next = (next_event.event_date - today).days

        return self._envelope(
            {
                "next_event": {
                    "date": next_event.iso_date,
                    "kind": next_event.kind,
                    "label": next_event.label,
                },
                "days_to_next_event": days_to_next,
                "days_by_kind": days_by_kind,
                "next_by_kind": next_by_kind,
            },
            Confidence.ok,
        )

    def min_days_to_event(self) -> int | None:
        """Convenience: return min days-to-next-event across kinds, or None."""

        env = self.get_calendar()
        if env.value is None:
            return None
        days_by_kind = env.value.get("days_by_kind") or {}
        if not days_by_kind:
            return None
        return min(days_by_kind.values())
