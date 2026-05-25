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
        help="Call Claude for verdict synthesis. Requires ANTHROPIC_API_KEY.",
    )
    analyze_p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the LLM model_version (default: price_table default_model).",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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
            model = args.model or price_table.get("default_model")
            try:
                from .llm import make_anthropic_client

                llm_client = make_anthropic_client(model=model)
            except RuntimeError as e:
                print(f"ERROR: --enable-llm but Anthropic SDK is unavailable: {e}", file=sys.stderr)
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

            sec_adapter = SECEdgarAdapter(registry)

        result = analyze(
            args.ticker.upper(),
            registry=registry,
            fred_adapter=fred_adapter,
            sec_edgar_adapter=sec_adapter,
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
