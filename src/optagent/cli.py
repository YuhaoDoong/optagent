"""Command-line entry point: `optagent analyze <ticker> [...]`.

Default mode is template_only (no LLM). `--enable-llm` switches to the
Claude tool_use synthesis path, gated by a deterministic budget pre-check
and validated by the AC-12 fail-closed validator.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .config_loader import load_bundle
from .orchestrator import analyze
from .registry import ProviderRegistry
from .profiles import ensure_default_profiles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optagent",
        description="US equity options research agent. RESEARCH ONLY — NOT FINANCIAL ADVICE.",
    )
    parser.add_argument("--version", action="version", version=f"optagent {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    screen_p = sub.add_parser(
        "screen",
        help="Run a strategy across a universe and recommend top candidates.",
    )
    screen_p.add_argument(
        "--strategy",
        type=str,
        default="oversold_rebound",
        help="Strategy id (default: oversold_rebound).",
    )
    screen_p.add_argument(
        "--universe",
        type=str,
        default="builtin",
        help="Either 'builtin' (default US large-cap list) or a path to a one-ticker-per-line file.",
    )
    screen_p.add_argument(
        "--sector",
        type=str,
        default=None,
        help=(
            "Intersect the universe with a sector before screening. "
            "See `optagent screen --list-sectors`."
        ),
    )
    screen_p.add_argument(
        "--list-sectors",
        action="store_true",
        help="Print the known sector keys and exit.",
    )
    screen_p.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top N to surface (default 5).",
    )
    screen_p.add_argument(
        "--min-market-cap",
        type=float,
        default=None,
        help="Soft filter (USD). Tickers below are dropped before scoring.",
    )
    screen_p.add_argument(
        "--min-avg-volume",
        type=float,
        default=None,
        help="Soft filter on 10-day avg share volume.",
    )

    analyze_p = sub.add_parser("analyze", help="Analyze a ticker and emit a research memo.")
    analyze_p.add_argument("ticker", help="US equity ticker (e.g. AAPL, SPY).")
    analyze_p.add_argument(
        "--horizon", type=int, default=14, help="Target holding horizon in days (default 14)."
    )
    analyze_p.add_argument(
        "--max-loss", type=float, default=None, help="Optional max-loss budget in USD."
    )
    analyze_p.add_argument(
        "--ledger-dir",
        type=Path,
        default=None,
        help="Override the audit ledger directory (default data/ledger).",
    )
    analyze_p.add_argument(
        "--no-ledger",
        action="store_true",
        help="Disable ledger writes (useful for one-off inspection).",
    )
    analyze_p.add_argument(
        "--enable-llm",
        action="store_true",
        help="Call an LLM for verdict synthesis.",
    )
    analyze_p.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=("anthropic", "openai", "gemini"),
        help="LLM provider. Auto-detected from env vars when omitted.",
    )
    analyze_p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model_version (default: provider's default).",
    )
    analyze_p.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Override the config directory (default: ./config).",
    )
    analyze_p.add_argument(
        "--no-fred",
        action="store_true",
        help="Skip the FRED macro adapter even if FRED_API_KEY is set.",
    )
    analyze_p.add_argument(
        "--no-sec",
        action="store_true",
        help="Skip the SEC EDGAR adapter (also disabled if no User-Agent set).",
    )
    analyze_p.add_argument(
        "--no-news",
        action="store_true",
        help="Skip the Yahoo News adapter even when yfinance is installed.",
    )
    analyze_p.add_argument(
        "--enable-ml",
        action="store_true",
        help=(
            "Train/use the per-ticker ML direction model (Alt-3 v0). First "
            "query for a ticker trains a fresh model (~5s); subsequent queries "
            "within %(default)s days hit the cache." % {"default": 7}
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    # Load gitignored .env (provider keys etc.) before reading os.environ.
    # Shell-exported vars always win over .env values.
    from .env_loader import load_dotenv

    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "screen":
        from . import DISCLAIMER
        from .strategies import (
            UniverseFilter,
            filter_to_sector,
            get_strategy,
            list_sectors,
            list_strategy_ids,
            load_universe,
            render_screen_report,
            screen_universe,
        )

        if args.list_sectors:
            print("Available sectors:")
            for s in list_sectors():
                print(f"  {s}")
            print("\nAvailable strategies:")
            for sid in list_strategy_ids():
                print(f"  {sid}")
            return 0

        try:
            strategy = get_strategy(args.strategy)
        except KeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

        universe = load_universe(args.universe)
        if args.sector is not None:
            try:
                universe = filter_to_sector(universe, args.sector)
            except KeyError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 2
            if not universe:
                print(
                    f"WARNING: sector '{args.sector}' has no overlap with the chosen universe.",
                    file=sys.stderr,
                )

        if args.min_market_cap is not None or args.min_avg_volume is not None:
            universe = UniverseFilter(
                min_market_cap_usd=args.min_market_cap,
                min_avg_volume=args.min_avg_volume,
            ).apply(universe)

        result = screen_universe(strategy, universe, top_n=args.limit)
        sys.stdout.write(render_screen_report(result, DISCLAIMER))
        return 0

    if args.command == "analyze":
        llm_client = None
        price_table = None
        ttl_table = None
        model = None

        if args.enable_llm:
            try:
                bundle = load_bundle(args.config_dir)
            except FileNotFoundError as e:
                print(f"ERROR: --enable-llm requires a config dir: {e}", file=sys.stderr)
                return 2
            price_table = bundle.price_table
            ttl_table = bundle.ttl_table
            try:
                from .llm import make_client_from_env

                llm_client, chosen_provider, model = make_client_from_env(
                    provider=args.provider, model=args.model
                )
                print(f"[llm] provider={chosen_provider} model={model}", file=sys.stderr)
            except RuntimeError as e:
                print(f"ERROR: --enable-llm: {e}", file=sys.stderr)
                return 2

        registry = ProviderRegistry()
        ensure_default_profiles(registry)

        fred_adapter = None
        if not args.no_fred and os.environ.get("FRED_API_KEY"):
            try:
                from .adapters import FREDAdapter

                fred_adapter = FREDAdapter(registry)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: FRED adapter disabled: {e}", file=sys.stderr)

        sec_adapter = None
        if not args.no_sec and os.environ.get("OPTAGENT_USER_AGENT"):
            from .adapters import SECEdgarAdapter
            from .adapters.sec_edgar_adapter import SECUserAgentMissingError

            try:
                sec_adapter = SECEdgarAdapter(registry)
            except SECUserAgentMissingError as e:
                print(f"WARNING: SEC adapter disabled: {e}", file=sys.stderr)

        news_adapter = None
        if not args.no_news:
            from .adapters import YahooNewsAdapter

            news_adapter = YahooNewsAdapter(registry)

        ml_adapter = None
        if args.enable_ml:
            from .ml import MLDirectionAdapter

            ml_adapter = MLDirectionAdapter()

        result = analyze(
            args.ticker.upper(),
            registry=registry,
            fred_adapter=fred_adapter,
            sec_edgar_adapter=sec_adapter,
            news_adapter=news_adapter,
            ml_direction_adapter=ml_adapter,
            horizon_days=args.horizon,
            max_loss_usd=args.max_loss,
            ledger_dir=args.ledger_dir,
            write_ledger=not args.no_ledger,
            enable_llm=args.enable_llm,
            llm_client=llm_client,
            model_version=model,
            price_table=price_table,
            ttl_table=ttl_table,
        )
        sys.stdout.write(result.memo)
        if result.ledger_path:
            print(f"\n[ledger] appended to {result.ledger_path}", file=sys.stderr)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
