"""Pluggable strategy framework for market-wide screening.

Each strategy implements `BaseStrategy.evaluate(...)` and returns either a
`StrategySignal` (a structured multi-timeframe observation) or `None`. The
screening CLI fans the strategy out across a curated universe, sorts by a
strategy-defined score, and surfaces the top N candidates.

Architecture intentionally mirrors the buy-side observation report
template the user provided: universe filter -> environment diagnosis
(multi-timeframe) -> pricing context -> friction -> reward/space -> notes
+ disclaimer.
"""

from .base import (
    DiagnosticBlock,
    FrictionBlock,
    PricingContextBlock,
    SignalDirection,
    StrategySignal,
    BaseStrategy,
)
from .oversold_rebound import OversoldRebound
from .registry import STRATEGY_REGISTRY, get_strategy, list_strategy_ids
from .screen import ScreenResult, render_screen_report, screen_universe
from .universe import (
    UniverseFilter,
    builtin_us_large_cap,
    load_universe,
)

__all__ = [
    "BaseStrategy",
    "DiagnosticBlock",
    "FrictionBlock",
    "PricingContextBlock",
    "SignalDirection",
    "StrategySignal",
    "OversoldRebound",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "list_strategy_ids",
    "ScreenResult",
    "render_screen_report",
    "screen_universe",
    "UniverseFilter",
    "builtin_us_large_cap",
    "load_universe",
]
