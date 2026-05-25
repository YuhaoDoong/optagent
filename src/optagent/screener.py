"""Deterministic contract screener.

Runs BEFORE any LLM call. Filters the raw chain by liquidity, DTE, and event
proximity; computes payoff math (`breakeven`, `max_loss`) and BS Greeks; ranks
the survivors. The LLM (when used) may only cite OCC symbols present in this
output.

Determinism: same input → byte-identical ranked candidate list (sort key:
liquidity_score DESC, days_to_event ASC, occ_symbol ASC).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import payoff, pricing
from .schemas import (
    OptionContract,
    OptionRight,
    RunConfig,
)


@dataclass(frozen=True)
class ScreenerThresholds:
    min_oi: int = 100
    min_volume: int = 10
    max_spread_pct: float = 0.25
    min_dte: int = 7
    max_dte: int = 45


@dataclass(frozen=True)
class ScreenerInputs:
    """Everything the screener needs from upstream adapters."""

    ticker: str
    spot: float
    rows: list[dict]
    expiration_str: str  # YYYY-MM-DD
    dte: int
    risk_free_rate: float  # decimal e.g. 0.045
    dividend_yield: float = 0.0
    days_to_event: int | None = None  # min(days_to_earnings, days_to_FOMC) etc.
    thresholds: ScreenerThresholds = ScreenerThresholds()


def _eval_one(
    row: dict,
    inp: ScreenerInputs,
    thresholds: ScreenerThresholds,
) -> tuple[OptionContract | None, str | None]:
    """Return (candidate, None) for accepted rows, (None, reason) for rejected."""

    occ = row.get("occ_symbol")
    if not occ:
        return None, "missing_field"

    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    oi = int(row.get("open_interest", 0) or 0)
    volume = int(row.get("volume", 0) or 0)
    strike = float(row.get("strike", 0.0) or 0.0)
    raw_iv = float(row.get("iv", 0.0) or 0.0)
    right = OptionRight(row["right"]) if isinstance(row.get("right"), str) else row["right"]

    if bid <= 0 or ask <= 0:
        return None, "missing_field"
    if strike <= 0:
        return None, "missing_field"

    sp = payoff.spread_pct(bid, ask)
    if sp == float("inf") or sp > thresholds.max_spread_pct:
        return None, "wide_spread"

    if oi < thresholds.min_oi:
        return None, "low_oi"
    if volume < thresholds.min_volume:
        return None, "low_volume"

    if inp.dte < thresholds.min_dte or inp.dte > thresholds.max_dte:
        return None, "dte_out_of_range"

    mid = payoff.mid_price(bid, ask)
    be = payoff.breakeven(right, strike, mid)
    ml = payoff.max_loss(mid)

    t_years = max(inp.dte / 365.0, 1e-6)
    # Use the broker-reported IV when it looks sane; otherwise solve for it.
    sigma = raw_iv if 0.01 < raw_iv < 5.0 else None
    if sigma is None:
        sigma_solved = pricing.implied_vol_brent(
            right, mid, inp.spot, strike, t_years, inp.risk_free_rate, inp.dividend_yield
        )
        if sigma_solved is None:
            return None, "missing_field"
        sigma = sigma_solved

    g = pricing.greeks(
        right, inp.spot, strike, t_years, inp.risk_free_rate, sigma, inp.dividend_yield
    )

    expiry_dt = datetime.strptime(inp.expiration_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    liquidity_score = _liquidity_score(oi, volume, sp)
    data_quality_score = _data_quality_score(bid, ask, raw_iv, sigma)

    contract = OptionContract(
        occ_symbol=occ,
        underlying=inp.ticker,
        expiration=expiry_dt,
        strike=strike,
        right=right,
        mid=round(mid, 4),
        bid=bid,
        ask=ask,
        spread_pct=round(sp, 4),
        oi=oi,
        volume=volume,
        delta=round(g["delta"], 4),
        theta=round(g["theta"], 4),
        vega=round(g["vega"], 4),
        iv=round(sigma, 4),
        breakeven=round(be, 4),
        max_loss=round(ml, 2),
        days_to_event=inp.days_to_event,
        liquidity_score=round(liquidity_score, 4),
        data_quality_score=round(data_quality_score, 4),
    )

    if inp.days_to_event is not None and 0 <= inp.days_to_event <= 1:
        return None, "event_too_close"

    return contract, None


def _liquidity_score(oi: int, volume: int, spread_pct: float) -> float:
    """Simple bounded score; higher = more liquid.

    Caps individual inputs so a single outlier can't dominate.
    """

    oi_term = min(oi / 5000.0, 1.0)  # saturates at 5000 OI
    vol_term = min(volume / 500.0, 1.0)  # saturates at 500/day
    spread_term = max(0.0, 1.0 - spread_pct / 0.25)  # 0 at 25% spread, 1 at tight
    return 0.45 * oi_term + 0.30 * vol_term + 0.25 * spread_term


def _data_quality_score(bid: float, ask: float, raw_iv: float, used_iv: float) -> float:
    """Penalises stale/crossed/locked/inconsistent quote artifacts."""

    score = 1.0
    if bid <= 0 or ask <= 0:
        score -= 0.4
    if ask < bid:
        score -= 0.4  # crossed
    if ask == bid:
        score -= 0.1  # locked
    if raw_iv <= 0 or raw_iv > 5.0:
        score -= 0.1
    if used_iv > 3.0:
        score -= 0.1  # tail-implausible
    return max(0.0, min(1.0, score))


@dataclass(frozen=True)
class ScreenerOutput:
    candidates: list[OptionContract]
    rejected: list[tuple[str, str]]  # (occ_symbol, rejection_reason)
    inputs_summary: dict


def screen(inp: ScreenerInputs, _run_config: RunConfig | None = None) -> ScreenerOutput:
    """Apply liquidity + DTE filters; rank surviving candidates.

    `_run_config` is reserved for future per-run threshold overrides; the v1
    screener uses the default `ScreenerThresholds`.
    """

    thr = inp.thresholds
    candidates: list[OptionContract] = []
    rejected: list[tuple[str, str]] = []

    for row in inp.rows:
        contract, reason = _eval_one(row, inp, thr)
        if contract is not None:
            candidates.append(contract)
        else:
            rejected.append((str(row.get("occ_symbol", "?")), reason or "missing_field"))

    # Deterministic sort: liquidity DESC, days_to_event ASC (None -> infinity), occ ASC.
    def _key(c: OptionContract):
        return (
            -c.liquidity_score,
            c.days_to_event if c.days_to_event is not None else 10_000_000,
            c.occ_symbol,
        )

    candidates.sort(key=_key)

    return ScreenerOutput(
        candidates=candidates,
        rejected=rejected,
        inputs_summary={
            "ticker": inp.ticker,
            "spot": inp.spot,
            "expiration": inp.expiration_str,
            "dte": inp.dte,
            "risk_free_rate": inp.risk_free_rate,
            "n_rows_in": len(inp.rows),
            "n_candidates": len(candidates),
            "n_rejected": len(rejected),
            "thresholds": {
                "min_oi": thr.min_oi,
                "min_volume": thr.min_volume,
                "max_spread_pct": thr.max_spread_pct,
                "min_dte": thr.min_dte,
                "max_dte": thr.max_dte,
            },
        },
    )


def split_by_bias(
    candidates: Iterable[OptionContract], bias: str
) -> tuple[OptionContract | None, OptionContract | None]:
    """Pick the most-liquid LONG_CALL and LONG_PUT candidate.

    Returns (best_call, best_put). When `bias='bullish'` callers should prefer
    the call; `bias='bearish'` → put; `bias='neutral'` → SKIP.
    """

    best_call = None
    best_put = None
    for c in candidates:
        if c.right is OptionRight.call and best_call is None:
            best_call = c
        elif c.right is OptionRight.put and best_put is None:
            best_put = c
        if best_call and best_put:
            break
    return best_call, best_put
