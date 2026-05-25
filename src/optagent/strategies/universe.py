"""Universe filtering — turn a candidate ticker list into a screening-ready
set with optional market-cap / options-volume gating.

v0.3 ships a curated US large-cap list (~50 mainstream tickers covering
most of the S&P 500's tradable options surface). For tighter universes
(S&P 500 / Russell 1000) callers can load a CSV via `load_universe()`.

`UniverseFilter` runs the optional caps over live yfinance fast_info; it
caches per-ticker market-cap lookups under `data/cache/universe_cache.json`
so repeated screens don't hammer Yahoo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


# Curated US large-cap universe for v0.3 screening. Hand-picked from the
# top-of-book by avg options volume to give the screener a sensible default
# starting set; users can override via `load_universe(...)`.
BUILTIN_US_LARGE_CAP: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "VTI",          # broad ETFs
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AVGO", "AMD", "NFLX",
    "ADBE", "CRM", "ORCL", "INTC", "QCOM",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    "JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY",
    "XOM", "CVX", "COP", "SLB",
    "HD", "WMT", "COST", "TGT", "LOW",
    "BA", "CAT", "DE", "GE", "HON",
    "DIS", "NKE", "MCD", "SBUX", "KO", "PEP",
    "T", "VZ", "CMCSA",
    "GLD", "SLV", "USO", "UNG",                 # commodities ETFs as macro proxies
)


def builtin_us_large_cap() -> list[str]:
    return list(BUILTIN_US_LARGE_CAP)


def load_universe(source: str | Path | Iterable[str]) -> list[str]:
    """Load a custom universe.

    - `"builtin"`: returns BUILTIN_US_LARGE_CAP.
    - A path to a CSV/text file: one ticker per line (blank lines + lines
      starting with '#' are skipped).
    - An iterable of strings: returned as a list.
    """

    if isinstance(source, str) and source.lower() == "builtin":
        return builtin_us_large_cap()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"universe file not found: {path}")
        out: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.upper())
        return out
    return [str(t).upper() for t in source]


# Sentinel that distinguishes "auto-detect yfinance" (the default) from
# "no yfinance available" (test path).
_AUTO = object()


@dataclass
class UniverseFilter:
    """Apply soft filters across a candidate ticker list.

    `min_market_cap_usd` and `min_avg_volume` are evaluated against
    `yfinance.Ticker(...).fast_info`; either can be `None` to disable.
    Pass `yf_module=None` explicitly to disable lookup (tests); leave it
    unset to auto-detect.
    Cache TTL is conservative (24h) since market cap moves slowly.
    """

    min_market_cap_usd: float | None = None
    min_avg_volume: float | None = None
    cache_path: Path | None = None
    cache_ttl_s: float = 86400.0
    yf_module: Any = _AUTO  # `None` means "explicitly disabled"

    def __post_init__(self) -> None:
        if self.yf_module is _AUTO:
            try:
                import yfinance as yf  # noqa: WPS433

                self.yf_module = yf
            except ImportError:
                self.yf_module = None
        if self.cache_path is None:
            self.cache_path = Path("data/cache/universe_cache.json")
        self._cache: dict[str, dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self.cache_path and self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _flush_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, separators=(",", ":")),
            encoding="utf-8",
        )

    def _lookup(self, ticker: str) -> dict[str, Any] | None:
        now = time.time()
        cached = self._cache.get(ticker)
        if cached and now - float(cached.get("fetched_at", 0)) < self.cache_ttl_s:
            return cached
        if self.yf_module is None:
            return None
        try:
            tk = self.yf_module.Ticker(ticker)
            info = tk.fast_info
            mkt_cap = (
                getattr(info, "market_cap", None)
                if info is not None
                else None
            )
            avg_vol = (
                getattr(info, "ten_day_average_volume", None)
                or getattr(info, "three_month_average_volume", None)
            )
            entry = {
                "market_cap": float(mkt_cap) if mkt_cap else None,
                "avg_volume": float(avg_vol) if avg_vol else None,
                "fetched_at": now,
            }
            self._cache[ticker] = entry
            self._flush_cache()
            return entry
        except Exception:  # noqa: BLE001 - degrade silently
            return None

    def apply(self, tickers: Iterable[str]) -> list[str]:
        """Return only tickers that pass every active filter.

        When the underlying lookup fails (no yfinance, network error), the
        ticker is KEPT (we don't have evidence to reject it). The strategy
        downstream still has its own checks.
        """

        result: list[str] = []
        for t in tickers:
            t = t.upper()
            entry = self._lookup(t)
            if entry is None:
                result.append(t)
                continue
            mc_ok = self.min_market_cap_usd is None or (
                entry.get("market_cap") is not None
                and entry["market_cap"] >= self.min_market_cap_usd
            )
            vol_ok = self.min_avg_volume is None or (
                entry.get("avg_volume") is not None
                and entry["avg_volume"] >= self.min_avg_volume
            )
            if mc_ok and vol_ok:
                result.append(t)
        return result
