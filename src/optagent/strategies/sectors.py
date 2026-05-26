"""Sector-mapped sub-universes.

The screening CLI can filter the universe to a single sector before
running a strategy, enabling news/policy-driven plays like
"oil prices spike → screen energy sector for momentum_breakout".

Sectors are hand-curated for v0.3 (mainstream US large-cap options
underlyings per GICS-style sector). Adjust via PR when issuers change.
"""

from __future__ import annotations


# Sector → ordered list of tickers.
# Each ticker should have a deep US options market.
SECTOR_TICKERS: dict[str, tuple[str, ...]] = {
    "broad_etf": ("SPY", "QQQ", "IWM", "DIA", "VTI"),
    "tech_mega": ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ADBE", "CRM", "ORCL"),
    # TSM is a foreign issuer ADR — not in BUILTIN_US_LARGE_CAP. Listed here
    # for completeness so callers passing a custom universe that includes
    # TSM still get the right sector mapping, but it'll be a no-op against
    # the built-in default universe.
    "tech_chips": ("NVDA", "AMD", "INTC", "AVGO", "QCOM", "TSM"),
    "communication": ("NFLX", "DIS", "CMCSA", "T", "VZ"),
    "financial": ("JPM", "BAC", "WFC", "GS", "MS", "V", "MA"),
    "healthcare": ("JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY"),
    "energy": ("XOM", "CVX", "COP", "SLB", "USO", "UNG"),
    "industrial": ("BA", "CAT", "DE", "GE", "HON"),
    "consumer_staples": ("WMT", "COST", "KO", "PEP"),
    "consumer_discretionary": ("HD", "TGT", "LOW", "NKE", "MCD", "SBUX", "TSLA"),
    "commodities_etf": ("GLD", "SLV", "USO", "UNG"),
}


def list_sectors() -> list[str]:
    return sorted(SECTOR_TICKERS.keys())


def tickers_for_sector(sector: str) -> list[str]:
    """Return ordered ticker list for a sector. Raises KeyError on unknown sector."""

    sector_key = sector.lower().strip()
    if sector_key not in SECTOR_TICKERS:
        raise KeyError(
            f"unknown sector {sector!r}; available: {list_sectors()}"
        )
    return list(SECTOR_TICKERS[sector_key])


def filter_to_sector(tickers: list[str], sector: str) -> list[str]:
    """Intersect a universe with a sector's tickers (preserving sector order)."""

    sector_set = set(tickers_for_sector(sector))
    tickers_set = set(t.upper() for t in tickers)
    return [t for t in tickers_for_sector(sector) if t in tickers_set]
