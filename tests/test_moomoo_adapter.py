"""Moomoo adapter — exercised with a fake OpenQuoteContext (no OpenD needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from optagent.adapters import MoomooAdapter
from optagent.adapters.moomoo_adapter import _occ_symbol
from optagent.profiles import ensure_default_profiles
from optagent.registry import ProviderRegistry
from optagent.schemas import Confidence, OptionRight, RunConfig


RET_OK = 0


class _FakeCtx:
    """Mimics the subset of OpenQuoteContext the adapter calls."""

    def __init__(self):
        self.closed = False

    def get_option_expiration_date(self, code):
        df = pd.DataFrame(
            [
                {"strike_time": "2026-06-03", "option_expiry_date_distance": 1},
                {"strike_time": "2026-06-10", "option_expiry_date_distance": 8},
            ]
        )
        return RET_OK, df

    def get_option_chain(self, code, start=None, end=None):
        df = pd.DataFrame(
            [
                {"code": "US.AAPL260610C00305000", "option_type": "CALL", "strike_price": 305.0},
                {"code": "US.AAPL260610P00300000", "option_type": "PUT", "strike_price": 300.0},
            ]
        )
        return RET_OK, df

    def get_market_snapshot(self, codes):
        rows = []
        for c in codes:
            if c.endswith("C00305000"):
                rows.append({
                    "code": c, "last_price": 5.95, "bid_price": 5.3, "ask_price": 6.35,
                    "volume": 112, "option_open_interest": 220, "option_implied_volatility": 26.834,
                    "option_strike_price": 305.0,
                })
            elif c.endswith("P00300000"):
                rows.append({
                    "code": c, "last_price": 2.5, "bid_price": 2.29, "ask_price": 2.66,
                    "volume": 1142, "option_open_interest": 1337, "option_implied_volatility": 27.56,
                    "option_strike_price": 300.0,
                })
        return RET_OK, pd.DataFrame(rows)

    def close(self):
        self.closed = True


def _entitled_registry():
    reg = ProviderRegistry()
    ensure_default_profiles(reg)
    reg.bind(RunConfig(ticker="AAPL", moomoo_entitled=True))
    return reg


def test_occ_symbol_format():
    assert _occ_symbol("AAPL", "2026-06-10", OptionRight.call, 305.0) == "AAPL260610C00305000"
    assert _occ_symbol("AAPL", "2026-06-10", OptionRight.put, 300.0) == "AAPL260610P00300000"


def test_chain_returns_real_quotes_and_converts_iv():
    reg = _entitled_registry()
    adapter = MoomooAdapter(reg, ctx=_FakeCtx())
    env = adapter.get_options_chain("AAPL", min_dte=7, max_dte=45)
    assert env.confidence is Confidence.ok
    assert env.source == "moomoo"
    assert env.value["expiration"] == "2026-06-10"
    assert env.value["dte"] == 8
    rows = {r["occ_symbol"]: r for r in env.value["rows"]}
    call = rows["AAPL260610C00305000"]
    assert call["bid"] == 5.3 and call["ask"] == 6.35
    assert call["open_interest"] == 220 and call["volume"] == 112
    # moomoo IV is percent -> fraction.
    assert abs(call["iv"] - 0.26834) < 1e-6
    assert call["right"] is OptionRight.call


def test_gate_blocks_without_entitlement():
    reg = ProviderRegistry()
    ensure_default_profiles(reg)
    reg.bind(RunConfig(ticker="AAPL", moomoo_entitled=False))  # default
    adapter = MoomooAdapter(reg, ctx=_FakeCtx())
    env = adapter.get_options_chain("AAPL")
    assert env.confidence is Confidence.unavailable
    assert "gate" in (env.warnings[0].lower() if env.warnings else "")


def test_unreachable_opend_is_unavailable_not_raise():
    reg = _entitled_registry()
    # ctx=None and SDK present but no real OpenD: _ensure_ctx tries to connect.
    # Force the path by injecting a ctx whose calls raise.
    class _BoomCtx:
        def get_option_expiration_date(self, code):
            raise ConnectionError("opend down")

    adapter = MoomooAdapter(reg, ctx=_BoomCtx())
    env = adapter.get_options_chain("AAPL")
    assert env.confidence is Confidence.unavailable


def test_get_price_from_snapshot():
    reg = _entitled_registry()

    class _PxCtx(_FakeCtx):
        def get_market_snapshot(self, codes):
            return RET_OK, pd.DataFrame([{"code": codes[0], "last_price": 306.31, "prev_close_price": 305.0}])

    adapter = MoomooAdapter(reg, ctx=_PxCtx())
    env = adapter.get_price("AAPL")
    assert env.confidence is Confidence.ok
    assert env.value["last"] == 306.31
