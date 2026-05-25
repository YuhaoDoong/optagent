"""Command-line entry point: `optagent analyze <ticker> [...]`.

Single subcommand for v1 release. Default mode is template_only (no LLM);
`--enable-llm` is reserved for a later round.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .orchestrator import analyze


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
        help="(Reserved) Enable LLM synthesis. Not wired in this release; ignored.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        if args.enable_llm:
            print(
                "WARNING: --enable-llm has no effect in this release; running in template_only mode.",
                file=sys.stderr,
            )
        result = analyze(
            args.ticker.upper(),
            horizon_days=args.horizon,
            max_loss_usd=args.max_loss,
            ledger_dir=args.ledger_dir,
            write_ledger=not args.no_ledger,
        )
        sys.stdout.write(result.memo)
        if result.ledger_path:
            print(f"\n[ledger] appended to {result.ledger_path}", file=sys.stderr)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
