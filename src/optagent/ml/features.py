"""Feature engineering for the ML direction model.

Pure-Python (no torch / no xgboost). The feature surface is small and
hand-picked so an adversarial pricing source can't easily corrupt the
target distribution.

Features (12):
  - ret_1d, ret_5d, ret_20d           — log returns over recent windows
  - rsi_14                            — relative strength index
  - macd_norm                         — MACD/spot, dimensionless
  - macd_signal_norm                  — MACD signal/spot
  - atr_14_pct                        — ATR(14) / close, % volatility proxy
  - dist_high_20                      — (close - 20d_high) / close
  - dist_low_20                       — (close - 20d_low) / close
  - vol_change_5d                     — volume(5d_avg) / volume(20d_avg)
  - hv20_annual                       — annualised 20d realised vol
  - hv60_annual                       — annualised 60d realised vol
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "rsi_14",
    "macd_norm",
    "macd_signal_norm",
    "atr_14_pct",
    "dist_high_20",
    "dist_low_20",
    "vol_change_5d",
    "hv20_annual",
    "hv60_annual",
)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the v0 feature matrix from an OHLCV DataFrame.

    Expects columns Open / High / Low / Close / Volume (yfinance shape).
    Output has one row per input row plus the FEATURE_NAMES columns; the
    first ~60 rows will contain NaN because of the rolling windows.
    """

    needed = {"Open", "High", "Low", "Close", "Volume"}
    missing = needed - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")

    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    volume = ohlcv["Volume"].astype(float)

    log_close = np.log(close.replace(0.0, np.nan))

    features = pd.DataFrame(index=ohlcv.index)
    features["ret_1d"] = log_close.diff(1)
    features["ret_5d"] = log_close.diff(5)
    features["ret_20d"] = log_close.diff(20)
    features["rsi_14"] = _rsi(close, 14)
    macd, signal = _macd(close)
    features["macd_norm"] = macd / close.replace(0.0, np.nan)
    features["macd_signal_norm"] = signal / close.replace(0.0, np.nan)
    features["atr_14_pct"] = _atr(high, low, close, 14) / close.replace(0.0, np.nan)
    features["dist_high_20"] = (close - close.rolling(20).max()) / close.replace(0.0, np.nan)
    features["dist_low_20"] = (close - close.rolling(20).min()) / close.replace(0.0, np.nan)
    vol5 = volume.rolling(5).mean()
    vol20 = volume.rolling(20).mean()
    features["vol_change_5d"] = vol5 / vol20.replace(0.0, np.nan)
    rets = log_close.diff()
    features["hv20_annual"] = rets.rolling(20).std() * math.sqrt(252)
    features["hv60_annual"] = rets.rolling(60).std() * math.sqrt(252)
    return features[list(FEATURE_NAMES)]


def build_target(
    close: pd.Series, horizon: int = 5, threshold: float = 0.0
) -> pd.Series:
    """Binary target: 1 if `close.shift(-horizon) / close` > 1 + threshold, else 0.

    Rows in the last `horizon` positions will be NaN (no future data).
    Calling code should drop NaN before fitting.
    """

    future = close.shift(-horizon)
    log_ret_h = np.log(future / close.replace(0.0, np.nan))
    target = (log_ret_h > threshold).astype(float)
    target.iloc[-horizon:] = np.nan
    return target
