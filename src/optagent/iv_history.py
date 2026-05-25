"""Per-ticker IV history store + IV-rank computation.

Accumulates a JSONL series of `{as_of, ticker, atm_iv_median, hv20_annual}`
rows under `data/iv_history/<TICKER>.jsonl`. The orchestrator calls
`append_snapshot()` after every successful run so the file grows over time.

`compute_iv_rank()` returns:
  - `None` when history is shorter than `min_observations` (default 30)
  - `float in [0, 100]` otherwise, where 50 means today's IV is the median
    of the trailing window

The screener consumes the result as informational context in v0.2 and may
gate on it once we accumulate enough history per ticker.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_HISTORY_DIR = Path("data/iv_history")
DEFAULT_MIN_OBSERVATIONS = 30
DEFAULT_WINDOW_OBSERVATIONS = 252  # ~1 trading year

# Strict ticker regex used BEFORE constructing any file path. Closes the
# path-traversal hole Codex R3 identified.
# Letters/digits, optionally followed by a SINGLE `.X` or `-X` suffix where X
# is also alphanumeric. Catches BRK.B / BF-B style symbols but rejects bare
# trailing dots (`AAPL..`) and consecutive separators.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,9}([.\-][A-Z0-9]{1,5})?$")

# Tolerated forward-clock skew when checking `as_of`. Anything beyond
# this counts as poisoned and is rejected.
_FUTURE_SKEW = timedelta(minutes=5)


class IVHistoryError(ValueError):
    """Raised for unsafe ticker symbols or other input violations."""


def _safe_ticker(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if not _TICKER_RE.match(t):
        raise IVHistoryError(f"unsafe_ticker_for_iv_history_path: {ticker!r}")
    return t


@dataclass(frozen=True)
class IVSnapshot:
    """A single per-ticker observation. JSONL-serialisable."""

    ticker: str
    as_of: datetime
    atm_iv_median: float
    hv20_annual: float | None = None

    @classmethod
    def from_json(cls, line: str) -> "IVSnapshot":
        d = json.loads(line)
        as_of = datetime.fromisoformat(d["as_of"].replace("Z", "+00:00"))
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        return cls(
            ticker=d["ticker"],
            as_of=as_of,
            atm_iv_median=float(d["atm_iv_median"]),
            hv20_annual=(float(d["hv20_annual"]) if d.get("hv20_annual") is not None else None),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "ticker": self.ticker,
                "as_of": self.as_of.astimezone(timezone.utc).isoformat(),
                "atm_iv_median": self.atm_iv_median,
                "hv20_annual": self.hv20_annual,
            },
            separators=(",", ":"),
        )


def _path_for(ticker: str, base: Path | None = None) -> Path:
    base = base or DEFAULT_HISTORY_DIR
    safe = _safe_ticker(ticker)
    return base / f"{safe}.jsonl"


def append_snapshot(
    ticker: str,
    *,
    atm_iv_median: float,
    hv20_annual: float | None,
    as_of: datetime | None = None,
    base: Path | None = None,
) -> Path | None:
    """Append a snapshot row. Returns the path written, or None if the input
    is non-finite (silently skip — we don't want to corrupt history on a
    bad day's data).
    """

    if not math.isfinite(atm_iv_median) or atm_iv_median <= 0:
        return None
    try:
        safe = _safe_ticker(ticker)
    except IVHistoryError:
        return None
    base = base or DEFAULT_HISTORY_DIR
    base.mkdir(parents=True, exist_ok=True)
    as_of_resolved = as_of or datetime.now(timezone.utc)
    # Reject future timestamps beyond a small clock-skew tolerance — closes
    # the timestamp-poisoning vector Codex R3 flagged.
    if as_of_resolved > datetime.now(timezone.utc) + _FUTURE_SKEW:
        return None
    # hv20 must be a finite positive number to be recorded; otherwise None.
    hv = None
    if (
        hv20_annual is not None
        and math.isfinite(hv20_annual)
        and hv20_annual > 0
    ):
        hv = float(hv20_annual)
    snap = IVSnapshot(
        ticker=safe,
        as_of=as_of_resolved,
        atm_iv_median=float(atm_iv_median),
        hv20_annual=hv,
    )
    path = _path_for(safe, base)
    with path.open("a", encoding="utf-8") as f:
        f.write(snap.to_json())
        f.write("\n")
    return path


def read_history(ticker: str, base: Path | None = None) -> list[IVSnapshot]:
    try:
        path = _path_for(ticker, base)
    except IVHistoryError:
        return []
    if not path.exists():
        return []
    out: list[IVSnapshot] = []
    cutoff = datetime.now(timezone.utc) + _FUTURE_SKEW
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                snap = IVSnapshot.from_json(line)
            except (ValueError, KeyError):
                continue
            # Ignore poisoned future-dated rows when reading back.
            if snap.as_of > cutoff:
                continue
            out.append(snap)
    return out


def compute_iv_rank(
    ticker: str,
    *,
    current_iv: float,
    base: Path | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    window: int = DEFAULT_WINDOW_OBSERVATIONS,
) -> dict | None:
    """Compute IV rank against the trailing window.

    Returns `None` when we don't have enough history to make the rank
    meaningful (so callers can fall back to IV/HV richness). Otherwise:
        {
            "rank_pct": <float 0..100>,
            "n_observations": <int>,
            "min": <float>,
            "max": <float>,
            "median": <float>,
            "window_start": <iso ts>,
            "window_end":   <iso ts>,
        }
    """

    if not math.isfinite(current_iv) or current_iv <= 0:
        return None
    history = read_history(ticker, base)
    if len(history) < min_observations:
        return None
    history.sort(key=lambda s: s.as_of)
    window_obs = history[-window:]
    vals = [s.atm_iv_median for s in window_obs if math.isfinite(s.atm_iv_median)]
    if len(vals) < min_observations:
        return None
    lo = min(vals)
    hi = max(vals)
    rng = hi - lo
    if rng <= 0:
        # All observations at the same level — surface rank=50 as neutral.
        rank_pct = 50.0
    else:
        rank_pct = 100.0 * (current_iv - lo) / rng
    rank_pct = max(0.0, min(100.0, rank_pct))
    vals_sorted = sorted(vals)
    median = vals_sorted[len(vals_sorted) // 2]
    return {
        "rank_pct": round(rank_pct, 2),
        "n_observations": len(vals),
        "min": round(lo, 4),
        "max": round(hi, 4),
        "median": round(median, 4),
        "window_start": window_obs[0].as_of.astimezone(timezone.utc).isoformat(),
        "window_end": window_obs[-1].as_of.astimezone(timezone.utc).isoformat(),
    }


def median_iv_from_chain_rows(rows: Iterable[dict]) -> float | None:
    """Median raw IV across chain rows whose IV looks sane.

    Used by the orchestrator to compute the snapshot value before appending.
    """

    sane: list[float] = []
    for r in rows:
        try:
            iv = float(r.get("iv", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if 0.01 < iv < 5.0 and math.isfinite(iv):
            sane.append(iv)
    if not sane:
        return None
    sane.sort()
    return sane[len(sane) // 2]
