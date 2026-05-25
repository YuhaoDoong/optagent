from __future__ import annotations

import pytest

from optagent.adapters.volume_oi_context import (
    VolumeOIContextAdapter,
    _max_pain,
    _wall,
    _pcr,
)
from optagent.profiles import ensure_default_profiles
from optagent.registry import ProviderRegistry
from optagent.schemas import Confidence, RunConfig


def _registry() -> ProviderRegistry:
    r = ProviderRegistry()
    ensure_default_profiles(r)
    r.bind(RunConfig(ticker="AAPL"))
    return r


def _row(occ: str, right: str, strike: float, oi: int, volume: int = 0) -> dict:
    return {
        "occ_symbol": occ,
        "right": right,
        "strike": strike,
        "open_interest": oi,
        "volume": volume,
        "bid": 1.0,
        "ask": 1.1,
    }


def test_max_pain_picks_strike_minimising_intrinsic():
    rows = [
        _row("C95", "call", 95, oi=100),
        _row("C100", "call", 100, oi=500),
        _row("C105", "call", 105, oi=100),
        _row("P95", "put", 95, oi=100),
        _row("P100", "put", 100, oi=500),
        _row("P105", "put", 105, oi=100),
    ]
    strike, pain = _max_pain(rows)
    # Symmetric chain ⇒ minimum pain at the centre strike.
    assert strike == 100
    assert pain == 1000.0  # (5*100 calls below 100 + 5*100 puts above) = 500+500


def test_wall_picks_strike_with_largest_oi():
    rows = [
        _row("C100", "call", 100, oi=500),
        _row("C105", "call", 105, oi=2000),
        _row("C110", "call", 110, oi=300),
    ]
    strike, oi = _wall(rows, "call")
    assert strike == 105
    assert oi == 2000


def test_pcr_returns_none_when_no_calls():
    rows = [_row("P100", "put", 100, oi=500)]
    assert _pcr(rows, "open_interest") is None


def test_compute_emits_caveat_and_signal():
    rows = [
        _row("C100", "call", 100, oi=500, volume=10),
        _row("P100", "put", 100, oi=600, volume=20),
    ]
    chain_value = {"expiration": "2026-06-19", "rows": rows}
    a = VolumeOIContextAdapter(_registry())
    env = a.compute(chain_value, spot=100.0)
    assert env.confidence is Confidence.ok
    assert env.value["caveat"].startswith("volume_oi_context is a derived")
    assert env.value["max_pain_strike"] == 100
    assert env.value["call_wall"]["strike"] == 100
    assert env.value["put_wall"]["strike"] == 100
    assert env.value["pcr_oi"] == pytest.approx(600 / 500)


def test_compute_unavailable_on_empty_chain():
    a = VolumeOIContextAdapter(_registry())
    env = a.compute({"rows": []}, spot=100.0)
    assert env.confidence is Confidence.unavailable
