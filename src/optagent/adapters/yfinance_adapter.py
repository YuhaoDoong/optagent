"""yfinance research-tier adapter.

yfinance is unofficial and labelled `research_only` (see provider profile);
quote data is typically delayed by ~15 minutes. The adapter:
  - wraps every output in an Envelope with `delay_assumption='delayed_15min'`,
  - never raises on missing data; returns `confidence=unavailable` instead,
  - calls into the upstream `yfinance` package only via deferred imports so
    test environments without the package can still import this module.

Network is touched only inside `_fetch_*` helpers; the public `get_*` methods
are unit-tested by injecting a fake `yfinance` module via the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from ..registry import ProviderRegistry
from ..schemas import (
    Confidence,
    Envelope,
    MarketSession,
    OptionRight,
)


YFINANCE_PROFILE_ID = "yfinance_research"


class YFinanceUnavailableError(RuntimeError):
    """Raised at adapter init when the yfinance package cannot be imported."""


@dataclass(frozen=True)
class _OptionRow:
    """Subset of yfinance option-chain row used by the screener."""

    occ_symbol: str
    strike: float
    right: OptionRight
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    iv: float


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _classify_session(now: datetime) -> MarketSession:
    """Cheap, timezone-agnostic session classifier.

    US equity options trade 09:30–16:00 ET. We do not import pytz; we just
    check the UTC hour against a wide RTH window (13:30–20:00 UTC during DST,
    14:30–21:00 outside DST). This is imprecise on DST transition days but
    good enough for envelope tagging — the screener never blindly trusts this
    flag, it also checks staleness.
    """

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


class YFinanceAdapter:
    """Adapter facade over the yfinance Python package.

    The constructor accepts an optional `yf_module` so tests can inject a
    fake. When `yf_module` is None the adapter attempts to import the real
    package lazily — and raises `YFinanceUnavailableError` if absent.
    """

    profile_id = YFINANCE_PROFILE_ID

    def __init__(
        self,
        registry: ProviderRegistry,
        yf_module: Any | None = None,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._registry = registry
        self._now = now
        if yf_module is not None:
            self._yf = yf_module
        else:
            try:
                import yfinance as yf  # noqa: WPS433
            except ImportError as e:
                raise YFinanceUnavailableError(
                    "yfinance is not installed; install with `pip install yfinance`"
                ) from e
            self._yf = yf

    # ------------------------------------------------------------------
    # Envelope helpers
    def _unavailable(self, reason: str) -> Envelope:
        now = self._now()
        return Envelope(
            value=None,
            as_of=now,
            source="yfinance",
            delay_assumption="delayed_15min",
            market_session=_classify_session(now),
            confidence=Confidence.unavailable,
            provider_profile_id=self.profile_id,
            warnings=[reason],
        )

    def _ok(self, value: Any, warnings: list[str] | None = None) -> Envelope:
        now = self._now()
        confidence = Confidence.degraded if warnings else Confidence.ok
        return Envelope(
            value=value,
            as_of=now,
            source="yfinance",
            delay_assumption="delayed_15min",
            market_session=_classify_session(now),
            confidence=confidence,
            provider_profile_id=self.profile_id,
            warnings=warnings or [],
        )

    def _check_gate(self) -> Envelope | None:
        gate = self._registry.gate(self.profile_id)
        if not gate.ok:
            return self._unavailable(f"compliance_gate_blocked: {gate.reason}")
        return None

    # ------------------------------------------------------------------
    # Public methods
    def get_price(self, ticker: str) -> Envelope:
        blocked = self._check_gate()
        if blocked is not None:
            return blocked
        try:
            tk = self._yf.Ticker(ticker)
            info = tk.fast_info  # yfinance >=0.2 attribute, dict-like
            last = info.get("last_price") if hasattr(info, "get") else None
            if last is None or last <= 0:
                return self._unavailable(f"no_last_price_for_{ticker}")
            return self._ok({"ticker": ticker, "last": float(last)})
        except Exception as e:  # broad on purpose: network/HTML/parse errors all degrade
            return self._unavailable(f"price_fetch_failed: {e.__class__.__name__}")

    def get_options_chain(
        self,
        ticker: str,
        min_dte: int = 7,
        max_dte: int = 45,
    ) -> Envelope:
        """Return the union of call/put rows for the first expiry within
        [min_dte, max_dte]. The screener consumes the unfiltered row list.
        """

        blocked = self._check_gate()
        if blocked is not None:
            return blocked
        if min_dte < 1 or max_dte < min_dte:
            return self._unavailable(f"bad_dte_window: min={min_dte} max={max_dte}")

        try:
            tk = self._yf.Ticker(ticker)
            expiries = list(tk.options)  # tuple/list of YYYY-MM-DD strings
        except Exception as e:
            return self._unavailable(f"chain_expiries_failed: {e.__class__.__name__}")

        if not expiries:
            return self._unavailable(f"no_expiries_for_{ticker}")

        today = self._now().date()
        target = None
        for s in expiries:
            try:
                d = datetime.strptime(s, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (d - today).days
            if min_dte <= dte <= max_dte:
                target = (s, dte)
                break

        if target is None:
            return self._unavailable(
                f"no_expiry_in_window: window=({min_dte},{max_dte}) expiries={expiries[:5]}"
            )

        expiry_str, dte = target

        try:
            chain = tk.option_chain(expiry_str)
        except Exception as e:
            return self._unavailable(f"option_chain_failed: {e.__class__.__name__}")

        calls = self._dataframe_to_rows(chain.calls, OptionRight.call)
        puts = self._dataframe_to_rows(chain.puts, OptionRight.put)
        rows = calls + puts

        if not rows:
            return self._unavailable(f"empty_chain_for_{ticker}_{expiry_str}")

        value = {
            "ticker": ticker,
            "expiration": expiry_str,
            "dte": dte,
            "rows": [r.__dict__ for r in rows],
        }
        return self._ok(value)

    # ------------------------------------------------------------------
    # Internal
    def _dataframe_to_rows(self, df: Any, right: OptionRight) -> list[_OptionRow]:
        """yfinance returns a pandas DataFrame; we extract a small column subset.

        We use `.itertuples()` to avoid a hard pandas import here.
        """

        out: list[_OptionRow] = []
        if df is None:
            return out
        # df.itertuples returns named tuples
        for row in df.itertuples(index=False):
            try:
                occ = str(getattr(row, "contractSymbol", ""))
                strike = float(getattr(row, "strike", 0.0))
                bid = float(getattr(row, "bid", 0.0) or 0.0)
                ask = float(getattr(row, "ask", 0.0) or 0.0)
                last_price = float(getattr(row, "lastPrice", 0.0) or 0.0)
                volume = int(getattr(row, "volume", 0) or 0)
                oi = int(getattr(row, "openInterest", 0) or 0)
                iv = float(getattr(row, "impliedVolatility", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if not occ or strike <= 0:
                continue
            out.append(
                _OptionRow(
                    occ_symbol=occ,
                    strike=strike,
                    right=right,
                    bid=bid,
                    ask=ask,
                    last_price=last_price,
                    volume=volume,
                    open_interest=oi,
                    iv=iv,
                )
            )
        return out
