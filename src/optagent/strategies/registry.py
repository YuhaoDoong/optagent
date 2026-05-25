"""Strategy registry — central lookup so CLI / docs share one source of truth."""

from __future__ import annotations

from typing import Callable, Type

from .base import BaseStrategy
from .oversold_rebound import OversoldRebound


STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    OversoldRebound.id: OversoldRebound,
}


def get_strategy(strategy_id: str) -> BaseStrategy:
    cls = STRATEGY_REGISTRY.get(strategy_id)
    if cls is None:
        raise KeyError(
            f"unknown strategy id {strategy_id!r}; available: {sorted(STRATEGY_REGISTRY.keys())}"
        )
    return cls()


def list_strategy_ids() -> list[str]:
    return sorted(STRATEGY_REGISTRY.keys())
