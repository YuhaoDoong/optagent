"""Strategy base class + the canonical `StrategySignal` data format.

Every strategy emits a `StrategySignal` so downstream tooling (CLI, ledger,
LLM prompt) consumes one stable schema. The five blocks mirror the
buy-side observation report layout:

  - environment (per-timeframe diagnostics)
  - pricing (IV / DTE / skew)
  - friction (spread / liquidity / halts / event silence)
  - reward (target price + potential repair space)
  - notes (free text for human-readable caveats)

Strategies decide their own scoring on top — the screener uses `score` to
rank candidates across a universe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

import pandas as pd

from .. import DISCLAIMER


class SignalDirection(str, Enum):
    """v0.3 keeps the bounded buy-side scope from v1 verdict."""

    long_call_observation = "long_call_observation"
    long_put_observation = "long_put_observation"
    skip = "skip"


@dataclass(frozen=True)
class DiagnosticBlock:
    """Per-timeframe environment diagnosis.

    `label` is the timeframe name ("daily", "60m", "15m" etc.). `conditions`
    is a dict of `{condition_name: bool|str|float}` so the audit ledger
    keeps the boolean rationale row that produced the verdict.
    """

    label: str
    summary: str
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PricingContextBlock:
    """Options-pricing context. Unknown fields are explicitly None."""

    iv_rank: float | None = None
    iv_percentile: float | None = None
    iv_skew_note: str | None = None
    dte_environment_note: str | None = None
    raw_chain_sampled: bool = False


@dataclass(frozen=True)
class FrictionBlock:
    """Execution friction check."""

    spread_pct: float | None = None
    liquidity_note: str | None = None
    halt_status: str = "normal"
    roll_status: str = "n/a"
    event_silence: str = "ok"


@dataclass(frozen=True)
class RewardBlock:
    """Reward / repair space."""

    target_price: float | None = None
    repair_space_pct: float | None = None
    rationale: str | None = None


@dataclass(frozen=True)
class StrategySignal:
    """Canonical strategy output. JSON-serialisable via `to_dict`.

    A strategy returns `direction=skip` when none of its conditions trigger;
    that allows the screener to keep the row in the audit ledger while
    excluding it from the recommended-list output.
    """

    strategy_id: str
    ticker: str
    timestamp: datetime
    spot: float
    direction: SignalDirection
    score: float  # higher = more compelling within this strategy
    daily: DiagnosticBlock
    hourly: DiagnosticBlock | None = None
    intraday: DiagnosticBlock | None = None
    pricing: PricingContextBlock = field(default_factory=PricingContextBlock)
    friction: FrictionBlock = field(default_factory=FrictionBlock)
    reward: RewardBlock = field(default_factory=RewardBlock)
    notes: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def __post_init__(self) -> None:
        # Enforce canonical disclaimer (Codex R4 finding). A strategy cannot
        # ship a signal with an empty or altered disclaimer.
        if self.disclaimer != DISCLAIMER:
            # `frozen=True` means we can't `self.disclaimer = DISCLAIMER`;
            # use object.__setattr__ to overwrite once, then it's frozen.
            object.__setattr__(self, "disclaimer", DISCLAIMER)

    def to_dict(self) -> dict[str, Any]:
        def _coerce(value: Any) -> Any:
            """Make values JSON-serialisable. Walks dicts + lists; coerces
            numpy scalars + bools via `.item()`."""

            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                return {k: _coerce(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_coerce(v) for v in value]
            # numpy scalars (np.bool_, np.float64, np.int64) all have .item().
            item = getattr(value, "item", None)
            if callable(item):
                try:
                    return item()
                except (TypeError, ValueError):
                    pass
            return str(value)

        def _opt(block):
            if block is None:
                return None
            return {k: _coerce(v) for k, v in block.__dict__.items()}

        return {
            "strategy_id": self.strategy_id,
            "ticker": self.ticker,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "spot": self.spot,
            "direction": self.direction.value,
            "score": round(self.score, 4),
            "daily": _opt(self.daily),
            "hourly": _opt(self.hourly),
            "intraday": _opt(self.intraday),
            "pricing": _opt(self.pricing),
            "friction": _opt(self.friction),
            "reward": _opt(self.reward),
            "notes": list(self.notes),
            "disclaimer": self.disclaimer,
        }


class BaseStrategy(ABC):
    """Abstract strategy interface.

    Concrete strategies set a class-level `id` and implement `evaluate()`.
    `evaluate` is called by the screener with the upstream OHLCV / chain
    data already fetched; the strategy should NOT touch the network.

    Returning None is allowed and equivalent to `direction=skip` but
    cheaper — the screener drops `None` rows from the ranked output.
    """

    id: ClassVar[str]

    @abstractmethod
    def evaluate(
        self,
        ticker: str,
        *,
        ohlcv_daily: pd.DataFrame,
        ohlcv_60m: pd.DataFrame | None = None,
        ohlcv_15m: pd.DataFrame | None = None,
        options_chain_value: dict | None = None,
        spot: float | None = None,
        now: datetime | None = None,
    ) -> StrategySignal | None:
        """Evaluate the strategy for a single ticker. Pure function."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "id", None):
            raise TypeError(f"strategy class {cls.__name__} must set a class-level `id`")
