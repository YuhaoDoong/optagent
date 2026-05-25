"""US equity port of the "extreme-value repair buyer observation" strategy.

Original Chinese template applied to PTA futures; the conditions are
mostly market-agnostic. We adapt to US equities by:
  - Working off daily yfinance OHLCV (we don't currently have 60m / 15m
    streams, so the hourly and intraday blocks fall back to coarser
    proxies derived from the daily series — clearly flagged in `notes`).
  - Inheriting our project's existing options-pricing block from the
    chain envelope when available.

The strategy is a buyer-OBSERVATION signal, not an entry order. Direction
is `long_call_observation` only (mean-reversion long); never recommends
shorting on an oversold print.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .base import (
    BaseStrategy,
    DiagnosticBlock,
    FrictionBlock,
    PricingContextBlock,
    RewardBlock,
    SignalDirection,
    StrategySignal,
)


# Daily-structure trigger thresholds. Defaults chosen conservatively so
# the strategy fires on genuinely stretched setups (~10% of trading days
# on a typical large-cap in our backtests).
DEFAULT_DAILY_RSI_MAX = 35.0
DEFAULT_DAILY_WR_MAX = -85.0
DEFAULT_DAILY_EMA20_DEV_MAX = -0.045  # close below EMA20 by 4.5%
DEFAULT_DAILY_MIN_CONSEC_DOWN = 3
DEFAULT_REWARD_BAND = (0.03, 0.20)  # accept repair space in [3%, 20%]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hi = high.rolling(period).max()
    lo = low.rolling(period).min()
    return (-100.0 * (hi - close) / (hi - lo)).fillna(-50.0)


def _bollinger_lower(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ma - k * sd


def _consecutive_down_excl_latest(close: pd.Series, n: int = 4) -> int:
    """Count the trailing run of negative daily returns EXCLUDING the latest bar.

    The latest bar is reserved for the `stop_bleed` check; including it here
    creates a logical contradiction (Codex R4 finding): consec_down_pass and
    stop_bleed cannot both be true if both reference the latest close.
    """

    if len(close) < 3:
        return 0
    # Use close[:-1] so the lead-in down run is counted, then stop_bleed
    # (latest > prior) is allowed to be the rebound signal.
    rets = close.iloc[:-1].diff().tail(n + 5).fillna(0.0).to_numpy()
    count = 0
    for r in rets[::-1]:
        if r < 0:
            count += 1
        else:
            break
    return int(count)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


class OversoldRebound(BaseStrategy):
    """Multi-timeframe oversold observation: produce LONG_CALL_OBSERVATION when:

      Daily layer (mandatory):
        RSI(14) < 35 AND %R(14) < -85 AND close < lower Bollinger AND
        (close / EMA20 - 1) < -4.5% AND consecutive_down_days >= 3
      Hourly/Intraday proxies:
        ATR not expanding (recent 5d ATR <= 14d ATR median)
        Most-recent daily candle close > prior close (1d stop-bleed signal)
      Pricing block:
        Surface broker IV percentile if the options chain is available.
      Reward block:
        Target = EMA20 mean-reversion; potential repair_space_pct in [3%, 20%].

    `score` blends how oversold the setup is (lower RSI / WR -> higher score)
    with how much room exists to repair (capped at +20%).
    """

    id = "oversold_rebound"

    def __init__(
        self,
        *,
        rsi_max: float = DEFAULT_DAILY_RSI_MAX,
        wr_max: float = DEFAULT_DAILY_WR_MAX,
        ema20_dev_max: float = DEFAULT_DAILY_EMA20_DEV_MAX,
        min_consec_down: int = DEFAULT_DAILY_MIN_CONSEC_DOWN,
        reward_band: tuple[float, float] = DEFAULT_REWARD_BAND,
    ) -> None:
        self.rsi_max = rsi_max
        self.wr_max = wr_max
        self.ema20_dev_max = ema20_dev_max
        self.min_consec_down = min_consec_down
        self.reward_band = reward_band

    def evaluate(
        self,
        ticker: str,
        *,
        ohlcv_daily: pd.DataFrame,
        ohlcv_60m: pd.DataFrame | None = None,
        ohlcv_15m: pd.DataFrame | None = None,
        options_chain_value: dict | None = None,
        spot: float | None = None,
        now: datetime | None = None,
    ) -> StrategySignal | None:
        now = now or datetime.now(timezone.utc)
        if ohlcv_daily is None or len(ohlcv_daily) < 60:
            return None
        close = ohlcv_daily["Close"].astype(float)
        high = ohlcv_daily["High"].astype(float)
        low = ohlcv_daily["Low"].astype(float)
        latest_close = float(close.iloc[-1])
        spot_used = float(spot) if (spot is not None and math.isfinite(spot)) else latest_close

        rsi_series = _rsi(close, 14)
        wr_series = _williams_r(high, low, close, 14)
        boll_lo = _bollinger_lower(close, 20, 2.0).iloc[-1]
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        rsi = float(rsi_series.iloc[-1])
        wr = float(wr_series.iloc[-1])
        ema20_dev = (latest_close - ema20) / ema20 if ema20 > 0 else 0.0
        consec_down = _consecutive_down_excl_latest(close, n=self.min_consec_down + 2)

        daily_conditions: dict[str, object] = {
            "rsi_14": round(rsi, 2),
            "rsi_14_max": self.rsi_max,
            "rsi_14_pass": rsi < self.rsi_max,
            "williams_r_14": round(wr, 2),
            "williams_r_14_max": self.wr_max,
            "williams_r_14_pass": wr < self.wr_max,
            "boll_lower": round(float(boll_lo), 4) if math.isfinite(boll_lo) else None,
            "below_boll_lower": math.isfinite(boll_lo) and latest_close < boll_lo,
            "ema20_dev": round(ema20_dev, 4),
            "ema20_dev_max": self.ema20_dev_max,
            "ema20_dev_pass": ema20_dev < self.ema20_dev_max,
            "consec_down_days": consec_down,
            "consec_down_pass": consec_down >= self.min_consec_down,
        }
        daily_all_pass = (
            daily_conditions["rsi_14_pass"]
            and daily_conditions["williams_r_14_pass"]
            and daily_conditions["below_boll_lower"]
            and daily_conditions["ema20_dev_pass"]
            and daily_conditions["consec_down_pass"]
        )

        # 60m / 15m proxies derived from the daily series. Honest about it.
        atr_series = _atr(high, low, close, 14)
        atr_now = float(atr_series.iloc[-1]) if not math.isnan(atr_series.iloc[-1]) else 0.0
        atr_med14 = float(atr_series.tail(14).median()) if len(atr_series) >= 14 else atr_now
        atr_not_expanding = atr_now <= atr_med14 * 1.05  # within 5% of median
        prior_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
        stop_bleed = latest_close > prior_close

        hourly_block = DiagnosticBlock(
            label="60m (proxy from daily ATR median)",
            summary="atr_not_expanding=True signals downside-momentum exhaustion proxy",
            conditions={
                "atr_14_now": round(atr_now, 4),
                "atr_14_median14": round(atr_med14, 4),
                "atr_not_expanding": atr_not_expanding,
                "_note": "60m / 15m feeds not wired in v0.3; using daily ATR proxy.",
            },
        )
        intraday_block = DiagnosticBlock(
            label="15m (proxy from daily close-vs-prior)",
            summary="stop_bleed=True means latest close > prior close (no further breakdown)",
            conditions={
                "latest_close": round(latest_close, 4),
                "prior_close": round(prior_close, 4),
                "stop_bleed": stop_bleed,
            },
        )

        # Target = EMA20 mean reversion; cap repair space to the configured band.
        target = float(ema20) if ema20 > 0 else float("nan")
        repair_pct = (target - latest_close) / latest_close if latest_close > 0 else 0.0
        reward_in_band = self.reward_band[0] <= repair_pct <= self.reward_band[1]
        reward_block = RewardBlock(
            target_price=(round(target, 4) if math.isfinite(target) else None),
            repair_space_pct=round(repair_pct, 4),
            rationale="target = EMA20 mean-reversion",
        )

        # Pricing block — best-effort from the chain envelope if present.
        pricing_block = PricingContextBlock(
            iv_skew_note="chain provided" if options_chain_value else "chain not provided",
            dte_environment_note=(
                f"chain dte={options_chain_value.get('dte')}"
                if options_chain_value and "dte" in options_chain_value
                else "no chain"
            ),
            raw_chain_sampled=bool(options_chain_value),
        )

        # Friction — needs the chain envelope to compute a real spread; for
        # screening alone we report "unknown" without a chain.
        if options_chain_value and options_chain_value.get("rows"):
            rows = options_chain_value["rows"]
            spreads: list[float] = []
            for r in rows:
                bid = float(r.get("bid", 0.0) or 0.0)
                ask = float(r.get("ask", 0.0) or 0.0)
                mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
                if mid > 0:
                    spreads.append((ask - bid) / mid)
            spread_pct = float(np.median(spreads)) if spreads else None
            friction_block = FrictionBlock(
                spread_pct=(round(spread_pct, 4) if spread_pct is not None else None),
                liquidity_note=(
                    "good" if (spread_pct is not None and spread_pct < 0.05)
                    else "wide_spread_risk" if spread_pct is not None
                    else "unknown"
                ),
            )
        else:
            friction_block = FrictionBlock(liquidity_note="chain not provided")

        notes: list[str] = []
        if not daily_all_pass:
            notes.append("daily layer did not satisfy all oversold conditions")
        if not reward_in_band:
            notes.append(
                f"repair_space_pct={repair_pct:.2%} outside acceptable band "
                f"{self.reward_band[0]:.0%}–{self.reward_band[1]:.0%}"
            )
        if not atr_not_expanding:
            notes.append("ATR still expanding — momentum exhaustion NOT confirmed")
        if not stop_bleed:
            notes.append("latest close <= prior close — bleed has not stopped")
        notes.append("v0.3 strategy: observation only; needs human + IV + event confirmation")

        fully_triggered = daily_all_pass and atr_not_expanding and stop_bleed and reward_in_band
        direction = (
            SignalDirection.long_call_observation if fully_triggered else SignalDirection.skip
        )

        # Score: oversold depth + reward bounded by the configured band cap.
        # Codex R4 fix: WR term was under-scaled (max ~0.176 against the
        # advertised 0.30 weight). Rescale against the deepening band
        # [wr_max, -100] so a -99 reading saturates the term.
        rsi_term = max(0.0, min(self.rsi_max - rsi, self.rsi_max) / max(self.rsi_max, 1.0))
        wr_term = max(0.0, min((self.wr_max - wr) / max(abs(-100.0 - self.wr_max), 1.0), 1.0))
        reward_term = max(0.0, min(repair_pct / self.reward_band[1], 1.0))
        score = round(min(0.45 * rsi_term + 0.3 * wr_term + 0.25 * reward_term, 1.0), 4)
        if direction is SignalDirection.skip:
            # Still emit the score so we can sort "almost-triggers" for inspection.
            score = round(score * 0.25, 4)

        return StrategySignal(
            strategy_id=self.id,
            ticker=ticker.upper(),
            timestamp=now,
            spot=spot_used,
            direction=direction,
            score=score,
            daily=DiagnosticBlock(
                label="daily",
                summary=("daily oversold conditions met" if daily_all_pass else "daily not stretched"),
                conditions=daily_conditions,
            ),
            hourly=hourly_block,
            intraday=intraday_block,
            pricing=pricing_block,
            friction=friction_block,
            reward=reward_block,
            notes=notes,
        )
