"""Replay harness — fixture-driven deterministic re-runs.

Provides:

  - `Clock`: callable returning a frozen `datetime`, used to make every
    `Envelope.fetched_at`, `Envelope.as_of` (when an adapter uses the
    clock), and `RunConfig.started_at` deterministic.

  - `record_fixture(...)`: convenience that wraps a list of real adapters,
    captures every Envelope they return into a JSON file.

  - `FixtureAdapters` (`build_replay_adapters(...)`): rebuilds adapter
    objects that pull from the fixture file instead of the network.

The goal is byte-stable template-mode output and schema-stable LLM-mode
output across runs for the same fixture + same prompt / model versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .schemas import Envelope


class Clock:
    """Frozen UTC clock used to keep envelope timestamps reproducible."""

    def __init__(self, iso_ts: str) -> None:
        # Accept "...Z" by replacing it with "+00:00" for fromisoformat.
        normalised = iso_ts.replace("Z", "+00:00")
        self._dt = datetime.fromisoformat(normalised)
        if self._dt.tzinfo is None:
            self._dt = self._dt.replace(tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._dt


# ---------------------------------------------------------------------------
# Fixture file shape


@dataclass(frozen=True)
class Fixture:
    """Wraps a fixture-on-disk representation of one run's upstream data."""

    ticker: str
    frozen_now: str  # ISO 8601 UTC
    yfinance_price: dict | None
    yfinance_history: dict | None
    yfinance_chain: dict | None
    econ_calendar: dict | None
    fred_macro: dict | None
    sec_recent_8k: dict | None

    @classmethod
    def load(cls, path: Path) -> "Fixture":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            ticker=payload["ticker"],
            frozen_now=payload["frozen_now"],
            yfinance_price=payload.get("yfinance_price"),
            yfinance_history=payload.get("yfinance_history"),
            yfinance_chain=payload.get("yfinance_chain"),
            econ_calendar=payload.get("econ_calendar"),
            fred_macro=payload.get("fred_macro"),
            sec_recent_8k=payload.get("sec_recent_8k"),
        )

    def dump(self, path: Path) -> None:
        payload = {
            "ticker": self.ticker,
            "frozen_now": self.frozen_now,
            "yfinance_price": self.yfinance_price,
            "yfinance_history": self.yfinance_history,
            "yfinance_chain": self.yfinance_chain,
            "econ_calendar": self.econ_calendar,
            "fred_macro": self.fred_macro,
            "sec_recent_8k": self.sec_recent_8k,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Fixture-backed adapter stubs


class _FixturedYFinance:
    """In-memory yfinance adapter substitute. Same surface as YFinanceAdapter."""

    profile_id = "yfinance_research"

    def __init__(self, fixture: Fixture, clock: Clock) -> None:
        self._fixture = fixture
        self._now = clock

    def _envelope(self, value: dict | None, source_kind: str) -> Envelope:
        from .schemas import Confidence, MarketSession

        now = self._now()
        if value is None:
            return Envelope(
                value=None,
                as_of=now,
                source="yfinance",
                delay_assumption="delayed_15min",
                market_session=MarketSession.rth,
                confidence=Confidence.unavailable,
                provider_profile_id=self.profile_id,
                warnings=[f"fixture_missing:{source_kind}"],
            )
        return Envelope(
            value=value,
            as_of=now,
            source="yfinance",
            delay_assumption="delayed_15min",
            market_session=MarketSession.rth,
            confidence=Confidence.ok,
            provider_profile_id=self.profile_id,
        )

    def get_price(self, ticker: str) -> Envelope:  # noqa: ARG002 - ticker fixed at fixture
        return self._envelope(self._fixture.yfinance_price, "price")

    def get_options_chain(self, ticker: str, min_dte: int = 7, max_dte: int = 45) -> Envelope:  # noqa: ARG002
        return self._envelope(self._fixture.yfinance_chain, "chain")

    def get_history(self, ticker: str, period: str = "60d") -> Envelope:  # noqa: ARG002
        return self._envelope(self._fixture.yfinance_history, "history")


class _FixturedEconCalendar:
    profile_id = "econ_calendar_builtin"

    def __init__(self, fixture: Fixture, clock: Clock) -> None:
        self._fixture = fixture
        self._now = clock

    def get_calendar(self) -> Envelope:
        from .schemas import Confidence, MarketSession

        now = self._now()
        value = self._fixture.econ_calendar
        if value is None:
            return Envelope(
                value=None,
                as_of=now,
                source="econ_calendar_builtin",
                delay_assumption="static_schedule",
                market_session=MarketSession.rth,
                confidence=Confidence.unavailable,
                provider_profile_id=self.profile_id,
                warnings=["fixture_missing:econ_calendar"],
            )
        return Envelope(
            value=value,
            as_of=now,
            source="econ_calendar_builtin",
            delay_assumption="static_schedule",
            market_session=MarketSession.rth,
            confidence=Confidence.ok,
            provider_profile_id=self.profile_id,
        )


@dataclass(frozen=True)
class FixtureAdapters:
    yfinance: _FixturedYFinance
    econ_calendar: _FixturedEconCalendar


def build_replay_adapters(fixture: Fixture) -> tuple[FixtureAdapters, Clock]:
    """Construct fixture-backed adapters and a clock frozen to `fixture.frozen_now`."""

    clock = Clock(fixture.frozen_now)
    return (
        FixtureAdapters(
            yfinance=_FixturedYFinance(fixture, clock),
            econ_calendar=_FixturedEconCalendar(fixture, clock),
        ),
        clock,
    )


# ---------------------------------------------------------------------------
# Capture helper


def capture_fixture(
    ticker: str,
    *,
    yfinance_adapter: Any | None,
    econ_calendar_adapter: Any | None = None,
    fred_adapter: Any | None = None,
    sec_edgar_adapter: Any | None = None,
    frozen_now: str | None = None,
    history_period: str = "60d",
) -> Fixture:
    """Capture upstream adapter outputs to a JSON-serialisable Fixture.

    Pass real adapters that have been already bound to a `ProviderRegistry`.
    The function calls each one once and unwraps the envelope `.value`.
    """

    now_iso = frozen_now or datetime.now(timezone.utc).isoformat()
    yf_price = yf_chain = yf_history = None
    if yfinance_adapter is not None:
        env = yfinance_adapter.get_price(ticker)
        yf_price = env.value
        env = yfinance_adapter.get_options_chain(ticker)
        yf_chain = env.value
        env = yfinance_adapter.get_history(ticker, period=history_period)
        yf_history = env.value
    econ = None
    if econ_calendar_adapter is not None:
        econ = econ_calendar_adapter.get_calendar().value
    fred = None
    if fred_adapter is not None:
        fred = fred_adapter.get_macro().value
    sec = None
    if sec_edgar_adapter is not None:
        sec = sec_edgar_adapter.get_recent_8k(ticker).value
    return Fixture(
        ticker=ticker,
        frozen_now=now_iso,
        yfinance_price=yf_price,
        yfinance_history=yf_history,
        yfinance_chain=yf_chain,
        econ_calendar=econ,
        fred_macro=fred,
        sec_recent_8k=sec,
    )


# ---------------------------------------------------------------------------
# Run replay


def replay(
    fixture_path: Path,
    *,
    ledger_dir: Path | None = None,
    write_ledger: bool = False,
) -> Any:
    """Replay a saved fixture through the orchestrator and return the result."""

    from .orchestrator import analyze
    from .profiles import ensure_default_profiles
    from .registry import ProviderRegistry
    from .schemas import RunConfig

    fixture = Fixture.load(fixture_path)
    adapters, clock = build_replay_adapters(fixture)

    registry = ProviderRegistry()
    ensure_default_profiles(registry)
    run_config = RunConfig(ticker=fixture.ticker, started_at=clock())
    registry.bind(run_config)

    return analyze(
        fixture.ticker,
        registry=registry,
        yfinance_adapter=adapters.yfinance,
        econ_calendar_adapter=adapters.econ_calendar,
        ledger_dir=ledger_dir,
        write_ledger=write_ledger,
    )
