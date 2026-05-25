"""Momentum breakout strategy — long-call observation.

Mirrors the OversoldRebound shape but for the opposite regime: a stock
that just broke above its recent range with rising volume and a strong
trend underneath. The point is to catch continuation, not chase
exhaustion — so we require the breakout to be:
  1) statistically meaningful (close above prior 20-day high),
  2) backed by volume expansion (5d avg vs 20d avg),
  3) on a still-healthy trend (close > EMA20 > EMA50 > 0 slope),
  4) with the momentum oscillators warm but not blown (RSI 55-78).

Reward target: a 1-ATR push from current spot (~5% on a typical large-cap)
capped to keep premium-buyer risk-reward sensible.
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


DEFAULT_RSI_MIN = 55.0
DEFAULT_RSI_MAX = 78.0
DEFAULT_MIN_VOLUME_RATIO = 1.20  # 5d / 20d > 1.2 = expansion
DEFAULT_MIN_EMA20_SLOPE_PCT = 0.001  # 0.1% per day rising
DEFAULT_BREAKOUT_LOOKBACK = 20
DEFAULT_REWARD_BAND = (0.02, 0.15)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


class MomentumBreakout(BaseStrategy):
    id = "momentum_breakout"

    def __init__(
        self,
        *,
        rsi_min: float = DEFAULT_RSI_MIN,
        rsi_max: float = DEFAULT_RSI_MAX,
        min_volume_ratio: float = DEFAULT_MIN_VOLUME_RATIO,
        min_ema20_slope_pct: float = DEFAULT_MIN_EMA20_SLOPE_PCT,
        breakout_lookback: int = DEFAULT_BREAKOUT_LOOKBACK,
        reward_band: tuple[float, float] = DEFAULT_REWARD_BAND,
    ) -> None:
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.min_volume_ratio = min_volume_ratio
        self.min_ema20_slope_pct = min_ema20_slope_pct
        self.breakout_lookback = breakout_lookback
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
        volume = ohlcv_daily["Volume"].astype(float)
        latest_close = float(close.iloc[-1])
        spot_used = float(spot) if (spot is not None and math.isfinite(spot)) else latest_close

        prior_high = float(
            close.iloc[-(self.breakout_lookback + 1) : -1].max()
            if len(close) > self.breakout_lookback
            else close.max()
        )
        broke_out = latest_close > prior_high

        vol5 = float(volume.tail(5).mean())
        vol20 = float(volume.tail(20).mean()) if len(volume) >= 20 else vol5
        vol_ratio = vol5 / vol20 if vol20 > 0 else 0.0

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema20_now = float(ema20.iloc[-1])
        ema50_now = float(ema50.iloc[-1])
        ema20_slope_pct = (
            (ema20_now / float(ema20.iloc[-5]) - 1.0)
            if len(ema20) >= 5 and float(ema20.iloc[-5]) > 0
            else 0.0
        )
        trend_healthy = latest_close > ema20_now > ema50_now and ema20_slope_pct >= self.min_ema20_slope_pct

        rsi_now = float(_rsi(close, 14).iloc[-1])
        rsi_warm = self.rsi_min <= rsi_now <= self.rsi_max

        daily_conditions: dict[str, object] = {
            "rsi_14": round(rsi_now, 2),
            "rsi_in_band": rsi_warm,
            "rsi_band": (self.rsi_min, self.rsi_max),
            "prior_high_lookback": self.breakout_lookback,
            "prior_high": round(prior_high, 4),
            "broke_out": broke_out,
            "vol_ratio_5d_over_20d": round(vol_ratio, 3),
            "vol_expansion": vol_ratio >= self.min_volume_ratio,
            "ema20": round(ema20_now, 4),
            "ema50": round(ema50_now, 4),
            "ema20_slope_pct_5d": round(ema20_slope_pct, 4),
            "trend_healthy": trend_healthy,
        }
        daily_pass = broke_out and rsi_warm and daily_conditions["vol_expansion"] and trend_healthy

        atr_series = _atr(high, low, close, 14)
        atr_now = float(atr_series.iloc[-1]) if not math.isnan(atr_series.iloc[-1]) else 0.0
        target = latest_close + atr_now if atr_now > 0 else latest_close * 1.03
        reward_pct = (target - latest_close) / latest_close if latest_close > 0 else 0.0
        reward_in_band = self.reward_band[0] <= reward_pct <= self.reward_band[1]

        hourly_block = DiagnosticBlock(
            label="60m (proxy from daily ATR)",
            summary="momentum continuation room = +1 ATR proxy",
            conditions={
                "atr_14": round(atr_now, 4),
                "atr_target": round(target, 4),
                "_note": "60m feed not wired; using daily ATR(14) as continuation room proxy",
            },
        )
        intraday_block = DiagnosticBlock(
            label="15m (proxy)",
            summary="no further-breakout confirmation feed in v0.3",
            conditions={"_note": "15m feed not wired; manual confirmation recommended"},
        )

        pricing_block = PricingContextBlock(
            iv_skew_note="chain provided" if options_chain_value else "chain not provided",
            dte_environment_note=(
                f"chain dte={options_chain_value.get('dte')}"
                if options_chain_value and "dte" in options_chain_value
                else "no chain"
            ),
            raw_chain_sampled=bool(options_chain_value),
        )
        friction_block = FrictionBlock(liquidity_note="check chain")

        notes: list[str] = []
        if not broke_out:
            notes.append("close has NOT broken above prior 20d high")
        if not rsi_warm:
            notes.append(
                f"RSI {rsi_now:.1f} outside warm band {self.rsi_min:.0f}-{self.rsi_max:.0f}"
            )
        if not daily_conditions["vol_expansion"]:
            notes.append(
                f"volume ratio {vol_ratio:.2f} below expansion threshold "
                f"{self.min_volume_ratio:.2f}"
            )
        if not trend_healthy:
            notes.append("EMA stack not bullish or EMA20 slope flat/down")
        if not reward_in_band:
            notes.append(
                f"reward_pct {reward_pct:.2%} outside band "
                f"{self.reward_band[0]:.0%}–{self.reward_band[1]:.0%}"
            )
        notes.append("v0.3 strategy: observation only; needs human + IV + event confirmation")

        fully_triggered = daily_pass and reward_in_band
        direction = (
            SignalDirection.long_call_observation if fully_triggered else SignalDirection.skip
        )

        rsi_term = max(0.0, min((rsi_now - self.rsi_min) / max(self.rsi_max - self.rsi_min, 1.0), 1.0))
        vol_term = max(0.0, min((vol_ratio - 1.0) / 1.0, 1.0))  # saturates at 2x volume
        reward_term = max(0.0, min(reward_pct / self.reward_band[1], 1.0))
        score = round(0.35 * rsi_term + 0.35 * vol_term + 0.30 * reward_term, 4)
        if direction is SignalDirection.skip:
            score *= 0.25

        return StrategySignal(
            strategy_id=self.id,
            ticker=ticker.upper(),
            timestamp=now,
            spot=spot_used,
            direction=direction,
            score=score,
            daily=DiagnosticBlock(
                label="daily",
                summary=("daily breakout + trend + volume confirmed" if daily_pass else "daily breakout NOT confirmed"),
                conditions=daily_conditions,
            ),
            hourly=hourly_block,
            intraday=intraday_block,
            pricing=pricing_block,
            friction=friction_block,
            reward=RewardBlock(
                target_price=round(target, 4),
                repair_space_pct=round(reward_pct, 4),
                rationale="target = close + 1×ATR(14)",
            ),
            notes=notes,
        )
