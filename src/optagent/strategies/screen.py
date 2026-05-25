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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from .base import BaseStrategy, SignalDirection, StrategySignal


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
    skipped: list[tuple[str, str]] = []
    n_evaluated = 0

    for ticker in universe:
        # Wrap the fetcher itself so one upstream error doesn't kill the
        # whole screen (Codex R4 finding — fetcher exceptions previously
        # escaped uncaught).
        try:
            ohlcv, chain = fetcher(ticker)
        except Exception as e:  # noqa: BLE001 - per-ticker robustness
            skipped.append((ticker, f"fetcher_raised:{e.__class__.__name__}"))
            continue
        if ohlcv is None:
            skipped.append((ticker, "no_ohlcv"))
            continue
        n_evaluated += 1
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
            # Still preserve the signal for audit but don't include in top-N
            continue
        triggered.append(signal)

    triggered.sort(key=lambda s: s.score, reverse=True)
    top = triggered[:top_n]
    return ScreenResult(
        strategy_id=strategy.id,
        universe_size=len(universe),
        n_evaluated=n_evaluated,
        n_triggered=len(triggered),
        top_signals=top,
        skipped=skipped,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
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
    if not result.top_signals:
        lines.append("Top candidates: (none triggered)")
        return "\n".join(lines) + "\n"
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
