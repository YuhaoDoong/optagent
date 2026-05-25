"""Data adapters. Every adapter call returns an Envelope (see schemas.py)."""

from .yfinance_adapter import (
    YFinanceAdapter,
    YFinanceUnavailableError,
)

__all__ = ["YFinanceAdapter", "YFinanceUnavailableError"]
