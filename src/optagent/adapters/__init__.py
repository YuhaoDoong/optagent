"""Data adapters. Every adapter call returns an Envelope (see schemas.py)."""

from .econ_calendar_adapter import EconCalendarAdapter, _Event as EconEvent
from .fred_adapter import FREDAdapter, FREDUnavailableError, SERIES_IDS as FRED_SERIES_IDS
from .sec_edgar_adapter import SECEdgarAdapter
from .yfinance_adapter import (
    YFinanceAdapter,
    YFinanceUnavailableError,
)

__all__ = [
    "EconCalendarAdapter",
    "EconEvent",
    "FREDAdapter",
    "FREDUnavailableError",
    "FRED_SERIES_IDS",
    "SECEdgarAdapter",
    "YFinanceAdapter",
    "YFinanceUnavailableError",
]
