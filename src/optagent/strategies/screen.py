"""Cross-ticker screening orchestrator.

`screen_universe()` fans a strategy out across a list of tickers, fetches
the minimum data each one needs (daily OHLCV and, when available, the
options chain), computes `StrategySignal` per ticker, and returns the
top-N by score among signals that fully triggered.

Designed to live alongside the per-ticker `analyze()` orchestrator —
they share the same disclaimers, audit-ledger row format, and downstream
LLM hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from .base import BaseStrategy, SignalDirection, StrategySignal


STALENESS_WARN_TRADING_DAYS = 2
"""Per Codex R5: surface a warning when the latest OHLCV bar lags the
current trading day by more than this many TRADING days. Calendar-day
logic false-positives on long weekends (Codex R5 case: Memorial Day
Mon 2026-05-25 left Tuesday with a 4-calendar-day-old Friday bar).
Using a Monday-Friday weekday count drops weekend skews."""


def _trading_days_between(start_date, end_date) -> int:
    """Count weekdays (Mon-Fri) strictly between start_date and end_date.

    Approximation — does NOT subtract US market holidays — but already
    handles the long-weekend false-positive that motivated the change.
    """

    if end_date <= start_date:
        return 0
    from datetime import timedelta

    count = 0
    cur = start_date + timedelta(days=1)
    while cur <= end_date:
        if cur.weekday() < 5:
            count += 1
        cur += timedelta(days=1)
    return count


@dataclass(frozen=True)
class ScreenResult:
    """Output of `screen_universe()`."""

    strategy_id: str
    universe_size: int
    n_evaluated: int
    n_triggered: int
    top_signals: list[StrategySignal]
    skipped: list[tuple[str, str]]  # (ticker, reason)
    started_at: datetime
    finished_at: datetime
    # Near-miss diagnostics — strategies that returned direction=skip but
    # scored above zero. Preserved per Codex R4 finding so the user can
    # inspect "almost triggers" without re-running.
    top_near_misses: list[StrategySignal] = field(default_factory=list)
    # Per-ticker stale-bar flags. Each entry is
    # (ticker, last_bar_iso_date, staleness_days).
    stale_bars: list[tuple[str, str, int]] = field(default_factory=list)


def make_default_fetcher(
    yf_module: Any | None = None,
) -> Callable[[str], tuple[pd.DataFrame | None, dict | None]]:
    """Public factory for the default per-ticker fetcher.

    Callers running several strategies over the same universe can build ONE
    fetcher (optionally memoized) and pass it to every `screen_universe` call so
    each ticker's history/chain is downloaded once instead of per strategy.
    """

    if yf_module is None:
        try:
            import yfinance as yf_module  # type: ignore  # noqa: WPS433
        except ImportError:
            yf_module = None
    return _default_fetcher(yf_module)


def _default_fetcher(yf_module: Any | None) -> Callable[[str], tuple[pd.DataFrame | None, dict | None]]:
    """Return a function that pulls (daily OHLCV, options_chain_value) per ticker."""

    if yf_module is None:
        return lambda t: (None, None)

    def _fetch(ticker: str) -> tuple[pd.DataFrame | None, dict | None]:
        try:
            tk = yf_module.Ticker(ticker)
            df = tk.history(period="6mo", interval="1d", auto_adjust=False)
            if df is None or df.empty:
                return None, None
            # Options chain best-effort; v0.3 screen doesn't strictly need it.
            chain = None
            try:
                expiries = list(tk.options or [])
                if expiries:
                    target = expiries[0]
                    ch = tk.option_chain(target)
                    rows = []
                    for side, frame in (("call", ch.calls), ("put", ch.puts)):
                        for row in frame.itertuples(index=False):
                            rows.append(
                                {
                                    "occ_symbol": getattr(row, "contractSymbol", ""),
                                    "right": side,
                                    "strike": float(getattr(row, "strike", 0.0)),
                                    "bid": float(getattr(row, "bid", 0.0) or 0.0),
                                    "ask": float(getattr(row, "ask", 0.0) or 0.0),
                                    "iv": float(getattr(row, "impliedVolatility", 0.0) or 0.0),
                                    "open_interest": int(getattr(row, "openInterest", 0) or 0),
                                    "volume": int(getattr(row, "volume", 0) or 0),
                                }
                            )
                    chain = {"expiration": target, "rows": rows}
            except Exception:  # noqa: BLE001 - degrade silently
                chain = None
            return df, chain
        except Exception:  # noqa: BLE001 - degrade silently
            return None, None

    return _fetch


def screen_universe(
    strategy: BaseStrategy,
    universe: list[str],
    *,
    fetcher: Callable[[str], tuple[pd.DataFrame | None, dict | None]] | None = None,
    yf_module: Any | None = None,
    top_n: int = 5,
    now: datetime | None = None,
) -> ScreenResult:
    """Run `strategy` across `universe`; return the top-N triggered signals."""

    now = now or datetime.now(timezone.utc)
    started_at = now
    if fetcher is None:
        if yf_module is None:
            try:
                import yfinance as yf  # noqa: WPS433

                yf_module = yf
            except ImportError:
                yf_module = None
        fetcher = _default_fetcher(yf_module)

    triggered: list[StrategySignal] = []
    near_misses: list[StrategySignal] = []
    skipped: list[tuple[str, str]] = []
    stale_bars: list[tuple[str, str, int]] = []
    n_evaluated = 0

    for ticker in universe:
        try:
            ohlcv, chain = fetcher(ticker)
        except Exception as e:  # noqa: BLE001 - per-ticker robustness
            skipped.append((ticker, f"fetcher_raised:{e.__class__.__name__}"))
            continue
        if ohlcv is None:
            skipped.append((ticker, "no_ohlcv"))
            continue
        n_evaluated += 1

        # Stale-bar diagnostic per Codex R4: a screen run on a US market
        # holiday silently consumes the prior trading day's bar. We don't
        # block, but we surface the staleness so the audit caller can warn.
        last_bar_dt = ohlcv.index[-1]
        if hasattr(last_bar_dt, "to_pydatetime"):
            last_bar_dt = last_bar_dt.to_pydatetime()
        if isinstance(last_bar_dt, datetime):
            last_bar_utc = (
                last_bar_dt.replace(tzinfo=timezone.utc)
                if last_bar_dt.tzinfo is None
                else last_bar_dt.astimezone(timezone.utc)
            )
            staleness_trading = _trading_days_between(last_bar_utc.date(), now.date())
            if staleness_trading >= STALENESS_WARN_TRADING_DAYS:
                stale_bars.append(
                    (ticker, last_bar_utc.date().isoformat(), staleness_trading)
                )

        try:
            signal = strategy.evaluate(
                ticker,
                ohlcv_daily=ohlcv,
                options_chain_value=chain,
                spot=float(ohlcv["Close"].iloc[-1]),
                now=now,
            )
        except Exception as e:  # noqa: BLE001 - one bad ticker shouldn't kill the screen
            skipped.append((ticker, f"strategy_raised:{e.__class__.__name__}"))
            continue
        if signal is None:
            skipped.append((ticker, "strategy_returned_none"))
            continue
        if signal.direction is SignalDirection.skip:
            # Preserve as a near-miss for diagnostics if it scored anything.
            if signal.score > 0:
                near_misses.append(signal)
            continue
        triggered.append(signal)

    triggered.sort(key=lambda s: s.score, reverse=True)
    near_misses.sort(key=lambda s: s.score, reverse=True)
    return ScreenResult(
        strategy_id=strategy.id,
        universe_size=len(universe),
        n_evaluated=n_evaluated,
        n_triggered=len(triggered),
        top_signals=triggered[:top_n],
        skipped=skipped,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        top_near_misses=near_misses[:top_n],
        stale_bars=stale_bars,
    )


def render_screen_report(result: ScreenResult, disclaimer: str) -> str:
    """Plain-text renderer with the same disclaimer-first convention as analyze()."""

    lines: list[str] = []
    lines.append(disclaimer)
    lines.append("=" * 72)
    lines.append(
        f"Strategy:      {result.strategy_id}"
    )
    lines.append(
        f"Universe size: {result.universe_size}    evaluated: {result.n_evaluated}    "
        f"triggered: {result.n_triggered}"
    )
    lines.append(
        f"Window:        {result.started_at.isoformat()}  ->  {result.finished_at.isoformat()}"
    )
    lines.append("")
    if result.stale_bars:
        lines.append("")
        lines.append(f"Stale-bar warnings ({len(result.stale_bars)}):")
        for ticker, iso_date, days in result.stale_bars[:10]:
            lines.append(f"  - {ticker}: last bar {iso_date} ({days}d behind today)")
        if len(result.stale_bars) > 10:
            lines.append(f"  - ... and {len(result.stale_bars) - 10} more")
    if not result.top_signals:
        lines.append("")
        lines.append("Top candidates: (none triggered)")
        if result.top_near_misses:
            lines.append("")
            lines.append("Top near-misses (direction=skip but partial trigger):")
            for i, sig in enumerate(result.top_near_misses[:5], start=1):
                lines.append(f"  {i}. {sig.ticker}  score={sig.score:.3f}")
        return "\n".join(lines) + "\n"
    lines.append("")
    lines.append("Top candidates:")
    for i, sig in enumerate(result.top_signals, start=1):
        lines.append(
            f"  {i}. {sig.ticker}  score={sig.score:.3f}  direction={sig.direction.value}"
            f"  spot=${sig.spot:.2f}"
        )
        if sig.reward.target_price is not None:
            lines.append(
                f"     target=${sig.reward.target_price:.2f}  "
                f"repair_space={sig.reward.repair_space_pct:+.2%}"
            )
        cond = sig.daily.conditions
        lines.append(
            f"     daily: RSI={cond.get('rsi_14')}  WR={cond.get('williams_r_14')}  "
            f"ema20_dev={cond.get('ema20_dev')}  consec_down={cond.get('consec_down_days')}"
        )
        for note in sig.notes:
            lines.append(f"     note: {note}")
    return "\n".join(lines) + "\n"
