"""Long-premium payoff math.

Breakeven and max_loss for long calls / long puts are pure accounting math,
NOT Black-Scholes outputs. Black-Scholes lives in `pricing.py` and is reserved
for Greeks and IV sanity checks.
"""

from __future__ import annotations

from .schemas import OptionRight


CONTRACT_MULTIPLIER = 100  # standard US equity options


def breakeven(right: OptionRight, strike: float, premium: float) -> float:
    """Per-share breakeven at expiration for a long-premium position."""

    if premium < 0:
        raise ValueError(f"premium must be non-negative, got {premium}")
    if right is OptionRight.call:
        return strike + premium
    return strike - premium


def max_loss(premium: float, contracts: int = 1) -> float:
    """Max-loss in USD for a long-premium position.

    Long premium has a bounded loss = premium paid (no margin / no naked risk).
    """

    if premium < 0:
        raise ValueError(f"premium must be non-negative, got {premium}")
    if contracts < 1:
        raise ValueError(f"contracts must be >= 1, got {contracts}")
    return premium * CONTRACT_MULTIPLIER * contracts


def spread_pct(bid: float, ask: float) -> float:
    """(ask - bid) / mid; 0.0 when mid is 0 or bid > ask."""

    if bid < 0 or ask < 0 or ask < bid:
        return float("inf")
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return float("inf")
    return (ask - bid) / mid


def mid_price(bid: float, ask: float) -> float:
    if bid < 0 or ask < 0 or ask < bid:
        raise ValueError(f"invalid bid/ask: bid={bid}, ask={ask}")
    return (bid + ask) / 2.0
