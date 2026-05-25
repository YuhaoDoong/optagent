#!/usr/bin/env python
"""Capture replay fixtures for a set of US-equity tickers.

Writes one JSON fixture per ticker into `tests/fixtures/<ticker>.json`.
Each fixture is a Replay `Fixture` payload containing the upstream adapter
output we'd otherwise re-fetch over the network. The fixtures power the
deterministic replay test suite (AC-6).

Usage:
    python scripts/capture_fixtures.py
    python scripts/capture_fixtures.py SPY QQQ AAPL --output-dir /tmp/fx
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


DEFAULT_TICKERS = ("SPY", "QQQ", "AAPL", "NVDA", "TSLA")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures",
        help="Directory to write fixture JSON files (default tests/fixtures/).",
    )
    parser.add_argument(
        "--include-sec",
        action="store_true",
        help="Also capture SEC EDGAR 8-K metadata (requires OPTAGENT_USER_AGENT).",
    )
    args = parser.parse_args()

    from optagent.adapters import (
        EconCalendarAdapter,
        SECEdgarAdapter,
        YFinanceAdapter,
    )
    from optagent.profiles import ensure_default_profiles
    from optagent.registry import ProviderRegistry
    from optagent.replay import capture_fixture
    from optagent.schemas import RunConfig

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_now = datetime.now(timezone.utc).isoformat()

    summaries: list[tuple[str, str]] = []
    for ticker in args.tickers:
        ticker = ticker.upper()
        registry = ProviderRegistry()
        ensure_default_profiles(registry)
        registry.bind(RunConfig(ticker=ticker))

        yfinance_adapter = YFinanceAdapter(registry)
        econ_adapter = EconCalendarAdapter(registry)
        sec_adapter = SECEdgarAdapter(registry) if args.include_sec else None

        fixture = capture_fixture(
            ticker,
            yfinance_adapter=yfinance_adapter,
            econ_calendar_adapter=econ_adapter,
            sec_edgar_adapter=sec_adapter,
            frozen_now=frozen_now,
        )
        out_path = args.output_dir / f"{ticker}.json"
        fixture.dump(out_path)
        rows = (fixture.yfinance_chain or {}).get("rows") if fixture.yfinance_chain else None
        n_rows = len(rows) if rows else 0
        summaries.append((ticker, f"{n_rows} chain rows -> {out_path.relative_to(REPO_ROOT)}"))

    print(f"\nCaptured {len(summaries)} fixture(s) at {frozen_now}")
    for t, s in summaries:
        print(f"  {t}: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
