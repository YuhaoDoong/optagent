from __future__ import annotations

from datetime import datetime, timezone

import pytest

from optagent.adapters.econ_calendar_adapter import EconCalendarAdapter, _Event
from optagent.profiles import ensure_default_profiles
from optagent.registry import ProviderRegistry
from optagent.schemas import Confidence, RunConfig, RunMode


def _registry(run_mode=RunMode.personal_research) -> ProviderRegistry:
    r = ProviderRegistry()
    ensure_default_profiles(r)
    r.bind(RunConfig(ticker="AAPL", run_mode=run_mode))
    return r


def _adapter(events=None, now=None) -> EconCalendarAdapter:
    return EconCalendarAdapter(
        _registry(),
        events=events,
        now=now or (lambda: datetime(2026, 6, 1, tzinfo=timezone.utc)),
    )


def test_calendar_picks_next_event():
    a = _adapter(
        events=[
            _Event("2026-06-10", "FOMC", "FOMC meeting"),
            _Event("2026-06-05", "NFP", "Nonfarm payrolls"),
            _Event("2026-06-15", "CPI", "CPI release"),
        ]
    )
    env = a.get_calendar()
    assert env.confidence is Confidence.ok
    assert env.value["next_event"]["kind"] == "NFP"
    assert env.value["next_event"]["date"] == "2026-06-05"
    assert env.value["days_to_next_event"] == 4


def test_days_by_kind_uses_first_occurrence():
    a = _adapter(
        events=[
            _Event("2026-06-05", "CPI", "x"),
            _Event("2026-06-10", "CPI", "y"),
        ]
    )
    env = a.get_calendar()
    assert env.value["days_by_kind"] == {"CPI": 4}


def test_min_days_to_event_returns_smallest():
    a = _adapter(
        events=[
            _Event("2026-06-10", "FOMC", "x"),
            _Event("2026-06-03", "CPI", "y"),
        ]
    )
    assert a.min_days_to_event() == 2


def test_calendar_empty_future_marks_degraded():
    a = _adapter(events=[_Event("2020-01-01", "FOMC", "ancient")])
    env = a.get_calendar()
    assert env.confidence is Confidence.degraded


def test_calendar_blocked_in_distributed_when_research_only():
    # econ_calendar_builtin profile is production_safe so distributed is fine —
    # confirm the gate doesn't block it (this is a regression guard).
    a = EconCalendarAdapter(_registry(run_mode=RunMode.distributed))
    env = a.get_calendar()
    assert env.confidence is not Confidence.unavailable
