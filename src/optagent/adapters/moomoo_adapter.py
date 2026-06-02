"""Moomoo (Futu) OpenD option-chain adapter.

Why this exists: the free yfinance/Yahoo feed zeroes out bid/ask AND
open-interest whenever the US market is closed (it keeps only last-trade
price + volume). The deterministic screener needs a two-sided quote and OI
to assess liquidity, so every after-hours run degrades to SKIP. Moomoo's
local OpenD gateway returns real bid/ask, open-interest, IV, and greeks
even at EOD — the same source the sibling `~/Gold` project uses.

Compliance: this is a personal-entitlement, research-only data path. It is
gated by the `moomoo_user_entitled` provider profile, which only passes when
`RunConfig.moomoo_entitled` is True. The adapter NEVER raises on a data
problem — it returns an `Envelope` with `confidence=unavailable` so the
orchestrator can fall back to yfinance and ultimately defer to SKIP.

Network/IPC is touched only inside the public `get_*` methods; tests inject
a fake quote context via the constructor so no OpenD is needed in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from ..registry import ProviderRegistry
from ..schemas import Confidence, Envelope, MarketSession, OptionRight


MOOMOO_PROFILE_ID = "moomoo_user_entitled"


class MoomooUnavailableError(RuntimeError):
    """Raised at adapter init when the moomoo SDK cannot be imported."""


@dataclass(frozen=True)
class _OptionRow:
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


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce a moomoo field to float; "N/A"/None/blank -> default."""

    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    # moomoo encodes missing numerics as NaN sometimes.
    return f if f == f else default  # f != f detects NaN


def _occ_symbol(ticker: str, expiry: str, right: OptionRight, strike: float) -> str:
    """Build a standard OCC-style symbol: ROOT + YYMMDD + C/P + strike*1000(8)."""

    y, m, d = expiry.split("-")
    cp = "C" if right is OptionRight.call else "P"
    strike_int = int(round(strike * 1000))
    return f"{ticker}{y[2:]}{m}{d}{cp}{strike_int:08d}"


class MoomooAdapter:
    """Adapter facade over the moomoo OpenQuoteContext.

    The constructor accepts an optional `ctx` so tests can inject a fake. When
    `ctx` is None the adapter connects to OpenD lazily on first use; if the
    SDK is missing or OpenD is unreachable, calls return `unavailable`
    envelopes rather than raising.
    """

    profile_id = MOOMOO_PROFILE_ID

    def __init__(
        self,
        registry: ProviderRegistry,
        ctx: Any | None = None,
        host: str = "127.0.0.1",
        port: int = 11111,
        now: Callable[[], datetime] = _utc_now,
        snapshot_batch: int = 200,
    ) -> None:
        self._registry = registry
        self._now = now
        self._host = host
        self._port = port
        self._ctx = ctx
        self._owns_ctx = ctx is None
        self._snapshot_batch = snapshot_batch

    # ------------------------------------------------------------------
    # Connection
    def _ensure_ctx(self) -> Any | None:
        if self._ctx is not None:
            return self._ctx
        try:
            import moomoo as ft  # noqa: WPS433
        except ImportError:
            try:
                import futu as ft  # type: ignore  # noqa: WPS433
            except ImportError:
                return None
        try:
            self._ctx = ft.OpenQuoteContext(host=self._host, port=self._port)
        except Exception:  # noqa: BLE001 — OpenD down / refused
            self._ctx = None
        return self._ctx

    def close(self) -> None:
        if self._ctx is not None and self._owns_ctx:
            try:
                self._ctx.close()
            except Exception:  # noqa: BLE001
                pass
            self._ctx = None

    @staticmethod
    def _ret_ok() -> int:
        try:
            import moomoo as ft  # noqa: WPS433

            return ft.RET_OK
        except ImportError:
            try:
                import futu as ft  # type: ignore  # noqa: WPS433

                return ft.RET_OK
            except ImportError:
                return 0

    # ------------------------------------------------------------------
    # Envelope helpers
    def _unavailable(self, reason: str) -> Envelope:
        now = self._now()
        return Envelope(
            value=None,
            as_of=now,
            source="moomoo",
            delay_assumption="realtime_entitled",
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
            source="moomoo",
            delay_assumption="realtime_entitled",
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
        ctx = self._ensure_ctx()
        if ctx is None:
            return self._unavailable("opend_unreachable")
        ok = self._ret_ok()
        try:
            ret, snap = ctx.get_market_snapshot([f"US.{ticker}"])
        except Exception as e:  # noqa: BLE001
            return self._unavailable(f"snapshot_failed: {e.__class__.__name__}")
        if ret != ok or snap is None or len(snap) == 0:
            return self._unavailable(f"no_price_for_{ticker}")
        last = _num(snap.iloc[0].get("last_price"))
        if last <= 0:
            last = _num(snap.iloc[0].get("prev_close_price"))
        if last <= 0:
            return self._unavailable(f"no_price_for_{ticker}")
        return self._ok({"ticker": ticker, "last": last})

    def get_options_chain(
        self,
        ticker: str,
        min_dte: int = 7,
        max_dte: int = 45,
    ) -> Envelope:
        blocked = self._check_gate()
        if blocked is not None:
            return blocked
        if min_dte < 1 or max_dte < min_dte:
            return self._unavailable(f"bad_dte_window: min={min_dte} max={max_dte}")
        ctx = self._ensure_ctx()
        if ctx is None:
            return self._unavailable("opend_unreachable")
        ok = self._ret_ok()
        code = f"US.{ticker}"

        # 1. Pick the first expiry inside the DTE window.
        try:
            ret, exp_df = ctx.get_option_expiration_date(code)
        except Exception as e:  # noqa: BLE001
            return self._unavailable(f"expiry_fetch_failed: {e.__class__.__name__}")
        if ret != ok or exp_df is None or len(exp_df) == 0:
            return self._unavailable(f"no_expiries_for_{ticker}")

        target = None
        for _i, r in exp_df.iterrows():
            dte = int(_num(r.get("option_expiry_date_distance"), -1))
            if min_dte <= dte <= max_dte:
                target = (str(r.get("strike_time"))[:10], dte)
                break
        if target is None:
            return self._unavailable(
                f"no_expiry_in_window: window=({min_dte},{max_dte})"
            )
        expiry_str, dte = target

        # 2. Pull the chain skeleton (codes + strikes + types) for that expiry.
        try:
            ret, chain = ctx.get_option_chain(code, start=expiry_str, end=expiry_str)
        except Exception as e:  # noqa: BLE001
            return self._unavailable(f"chain_fetch_failed: {e.__class__.__name__}")
        if ret != ok or chain is None or len(chain) == 0:
            return self._unavailable(f"empty_chain_for_{ticker}_{expiry_str}")

        codes = [str(c) for c in chain["code"].tolist()]

        # 3. Snapshot the codes in batches for quotes / OI / IV.
        snaps: dict[str, Any] = {}
        for i in range(0, len(codes), self._snapshot_batch):
            batch = codes[i : i + self._snapshot_batch]
            try:
                ret, snap = ctx.get_market_snapshot(batch)
            except Exception as e:  # noqa: BLE001
                return self._unavailable(f"snapshot_failed: {e.__class__.__name__}")
            if ret != ok or snap is None:
                continue
            for _j, srow in snap.iterrows():
                snaps[str(srow.get("code"))] = srow

        rows: list[_OptionRow] = []
        for _i, crow in chain.iterrows():
            ccode = str(crow.get("code"))
            srow = snaps.get(ccode)
            if srow is None:
                continue
            raw_type = str(crow.get("option_type", "")).upper()
            right = OptionRight.call if raw_type == "CALL" else (
                OptionRight.put if raw_type == "PUT" else None
            )
            if right is None:
                continue
            strike = _num(crow.get("strike_price")) or _num(srow.get("option_strike_price"))
            if strike <= 0:
                continue
            # moomoo IV is in PERCENT (e.g. 26.834 -> 0.26834).
            iv = _num(srow.get("option_implied_volatility")) / 100.0
            rows.append(
                _OptionRow(
                    occ_symbol=_occ_symbol(ticker, expiry_str, right, strike),
                    strike=strike,
                    right=right,
                    bid=_num(srow.get("bid_price")),
                    ask=_num(srow.get("ask_price")),
                    last_price=_num(srow.get("last_price")),
                    volume=int(_num(srow.get("volume"))),
                    open_interest=int(_num(srow.get("option_open_interest"))),
                    iv=iv,
                )
            )

        if not rows:
            return self._unavailable(f"no_quotable_rows_for_{ticker}_{expiry_str}")

        value = {
            "ticker": ticker,
            "expiration": expiry_str,
            "dte": dte,
            "rows": [r.__dict__ for r in rows],
        }
        return self._ok(value)
