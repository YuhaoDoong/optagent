"""Pure presentation helpers — no Streamlit state, no network.

Keeping these as pure functions means they're unit-testable without
Streamlit's TestClient. The Streamlit `app.py` is a thin orchestration
layer that calls these helpers to build pages.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import pandas as pd

from .. import DISCLAIMER
from ..schemas import Envelope, OptionContract, Verdict, VerdictAction


DISCLAIMER_BANNER = DISCLAIMER


# Direction → (label, color) — keep monochrome/safe palette; UI shouldn't
# look like a "buy this" signal.
_DIRECTION_STYLE: dict[str, tuple[str, str]] = {
    "SKIP": ("SKIP", "#6b7280"),
    "LONG_CALL": ("LONG CALL (observation)", "#0ea5e9"),
    "LONG_PUT": ("LONG PUT (observation)", "#a78bfa"),
    # Strategy directions (different enum):
    "long_call_observation": ("Long-call observation", "#0ea5e9"),
    "long_put_observation": ("Long-put observation", "#a78bfa"),
    "skip": ("Skip", "#6b7280"),
}


def verdict_badge(verdict: Verdict) -> dict[str, Any]:
    """Return a small dict the UI can use to render a coloured verdict badge.

    No Streamlit dependency here; the UI layer turns the dict into markup.
    """

    label, color = _DIRECTION_STYLE.get(
        verdict.action.value, (verdict.action.value, "#374151")
    )
    skip_reason = verdict.skip_reason.value if verdict.skip_reason else None
    return {
        "label": label,
        "color": color,
        "action": verdict.action.value,
        "conviction": verdict.conviction,
        "skip_reason": skip_reason,
        "disclaimer": verdict.disclaimer,
    }


def candidate_table(candidates: Iterable[OptionContract]) -> pd.DataFrame:
    """Flatten screener candidates into a tidy DataFrame for `st.dataframe`."""

    rows = []
    for c in candidates:
        rows.append(
            {
                "OCC": c.occ_symbol,
                "Right": c.right.value,
                "Strike": c.strike,
                "Mid": c.mid,
                "Spread %": c.spread_pct,
                "OI": c.oi,
                "Vol": c.volume,
                "Δ": c.delta,
                "θ/day": c.theta,
                "ν": c.vega,
                "IV": c.iv,
                "Breakeven": c.breakeven,
                "Max-loss $": c.max_loss,
                "Liq score": c.liquidity_score,
                "DQ score": c.data_quality_score,
            }
        )
    return pd.DataFrame(rows)


def envelope_summary(envelopes: Iterable[Envelope]) -> pd.DataFrame:
    rows = []
    for e in envelopes:
        rows.append(
            {
                "source": e.source,
                "profile": e.provider_profile_id,
                "confidence": e.confidence.value,
                "as_of": e.as_of.isoformat(),
                "delay": e.delay_assumption,
                "session": e.market_session.value,
                "cache_age_s": e.cache_age_s,
                "warnings": ", ".join(e.warnings) if e.warnings else "",
            }
        )
    return pd.DataFrame(rows)


def ml_signal_gauge(ml_signal: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compute the gauge inputs for `prob_up` (range [0, 1]).

    Returns a dict the UI can hand to a Plotly indicator: `value`, `range`,
    `threshold`, `subtitle`, plus a credibility text annotation.
    """

    if ml_signal is None:
        return {"available": False}
    prob = ml_signal.get("prob_up")
    if prob is None:
        return {"available": False}
    credibility = ml_signal.get("credibility", "low")
    ci_lo = ml_signal.get("wilson_ci_lower")
    ci_hi = ml_signal.get("wilson_ci_upper")
    baseline = ml_signal.get("class_baseline_accuracy")
    n_samples = ml_signal.get("n_oos_samples")
    return {
        "available": True,
        "prob_up": float(prob),
        "class_label": ml_signal.get("class_label"),
        "credibility": credibility,
        "subtitle": (
            f"OOS acc {ml_signal.get('oos_accuracy')} | "
            f"CI 95% [{ci_lo}, {ci_hi}] | "
            f"baseline {baseline} | "
            f"n={n_samples} | "
            f"credibility={credibility}"
        ),
        "feature_snapshot": ml_signal.get("feature_snapshot") or {},
    }


def feature_radar(feature_snapshot: Mapping[str, float] | None) -> pd.DataFrame:
    """Tidy DataFrame for a radar/spider chart of the ML feature snapshot.

    We normalise each feature into roughly [-1, 1] so the radar shape is
    interpretable without context. Hard-coded scales reflect typical
    cross-section magnitudes — close enough for visualisation.
    """

    if not feature_snapshot:
        return pd.DataFrame(columns=["feature", "value", "normalised"])

    scales: dict[str, float] = {
        "ret_1d": 0.05,
        "ret_5d": 0.15,
        "ret_20d": 0.30,
        "rsi_14": 50.0,  # /50 -> centered around 1.0; UI subtracts 1.0
        "macd_norm": 0.05,
        "macd_signal_norm": 0.05,
        "atr_14_pct": 0.05,
        "dist_high_20": 0.10,
        "dist_low_20": 0.10,
        "vol_change_5d": 2.0,  # /2.0 centered around 0.5
        "hv20_annual": 0.50,
        "hv60_annual": 0.50,
    }
    rows = []
    for name, val in feature_snapshot.items():
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        scale = scales.get(name, 1.0)
        if name in {"rsi_14"}:
            norm = (val - 50.0) / 50.0  # centred at 0
        elif name in {"vol_change_5d"}:
            norm = (val - 1.0) / 1.0  # 1.0 == no change
        else:
            norm = val / scale if scale > 0 else val
        norm = max(-1.0, min(1.0, norm))
        rows.append({"feature": name, "value": round(val, 4), "normalised": round(norm, 4)})
    return pd.DataFrame(rows)


def candle_chart(ohlcv: pd.DataFrame, *, max_rows: int = 60) -> pd.DataFrame:
    """Slice the last `max_rows` of an OHLCV frame and add some overlays."""

    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()
    tail = ohlcv.tail(max_rows).copy()
    if "Close" in tail.columns:
        tail["EMA20"] = tail["Close"].astype(float).ewm(span=20, adjust=False).mean()
        tail["EMA50"] = tail["Close"].astype(float).ewm(span=50, adjust=False).mean()
    return tail


def iv_smile_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten chain rows into a (strike, right, iv) frame for the IV-smile chart.

    Filters rows whose iv is finite + within [0.01, 5.0] (the same sanity
    range the screener applies). Sorts by strike per right so plotly draws
    smooth curves.
    """

    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            iv = float(r.get("iv", 0.0) or 0.0)
            strike = float(r.get("strike", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not (0.01 < iv < 5.0) or strike <= 0:
            continue
        right = r.get("right")
        if right not in ("call", "put"):
            continue
        out.append({"strike": strike, "iv": iv, "right": right})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["right", "strike"]).reset_index(drop=True)


def ledger_index(ledger_dir: Any, days_back: int = 7) -> pd.DataFrame:
    """Summarise recent JSONL ledger rows for the ledger viewer.

    Returns a DataFrame with one row per run: run_id, ticker, run_mode,
    verdict_action, skip_reason, started_at, finished_at, n_envelopes,
    n_screener_candidates. Pure stdlib + pandas — no Streamlit dep.

    `ledger_dir` is anything accepted by pathlib.Path.
    """

    import json
    from datetime import date as _date, datetime as _dt, timedelta, timezone as _tz
    from pathlib import Path

    base = Path(ledger_dir)
    if not base.exists():
        return pd.DataFrame()

    today = _dt.now(_tz.utc).date()
    cutoff = today - timedelta(days=max(days_back, 1) - 1)
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.jsonl")):
        try:
            file_date = _date.fromisoformat(path.stem)
        except ValueError:
            continue
        if file_date < cutoff or file_date > today:
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                verdict = rec.get("final_verdict") or {}
                rows.append(
                    {
                        "started_at": rec.get("started_at"),
                        "ticker": rec.get("ticker"),
                        "run_mode": rec.get("run_mode"),
                        "action": verdict.get("action"),
                        "skip_reason": verdict.get("skip_reason"),
                        "n_envelopes": len(rec.get("envelopes") or []),
                        "n_candidates": len(rec.get("screener_output") or []),
                        "run_id": rec.get("run_id"),
                        "ledger_file": path.name,
                    }
                )
        except OSError:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("started_at", ascending=False).reset_index(drop=True)
    return df


def strategy_signal_table(signals: list[Any]) -> pd.DataFrame:
    """Flatten a list of StrategySignal into a tidy table for the screen page."""

    rows = []
    for s in signals:
        d = s.to_dict() if hasattr(s, "to_dict") else dict(s)
        reward = d.get("reward") or {}
        daily = d.get("daily") or {}
        cond = daily.get("conditions") if isinstance(daily, dict) else {}
        cond = cond or {}
        rows.append(
            {
                "Ticker": d.get("ticker"),
                "Direction": d.get("direction"),
                "Score": d.get("score"),
                "Spot": d.get("spot"),
                "Target": reward.get("target_price"),
                "Repair %": reward.get("repair_space_pct"),
                "RSI": cond.get("rsi_14"),
                "WR": cond.get("williams_r_14"),
                "EMA20 dev": cond.get("ema20_dev"),
                "Notes": "; ".join((d.get("notes") or [])[:2]),
            }
        )
    return pd.DataFrame(rows)
