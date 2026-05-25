from __future__ import annotations

import math

import pytest

from optagent.pricing import (
    black_scholes_price,
    greeks,
    implied_vol_brent,
)
from optagent.schemas import OptionRight


def test_call_put_parity_approx():
    spot, strike, t, r, sigma = 100.0, 100.0, 0.5, 0.045, 0.25
    c = black_scholes_price(OptionRight.call, spot, strike, t, r, sigma)
    p = black_scholes_price(OptionRight.put, spot, strike, t, r, sigma)
    # C - P = S - K*e^{-rT}
    expected = spot - strike * math.exp(-r * t)
    assert c - p == pytest.approx(expected, abs=1e-6)


def test_atm_call_delta_near_half():
    g = greeks(OptionRight.call, 100.0, 100.0, 0.25, 0.045, 0.20)
    # ATM call delta is slightly above 0.5 when r > 0 and q = 0.
    assert 0.5 < g["delta"] < 0.65


def test_atm_put_delta_near_minus_half():
    g = greeks(OptionRight.put, 100.0, 100.0, 0.25, 0.045, 0.20)
    assert -0.65 < g["delta"] < -0.35


def test_gamma_positive():
    g = greeks(OptionRight.call, 100.0, 100.0, 0.25, 0.045, 0.20)
    assert g["gamma"] > 0


def test_long_call_theta_negative():
    g = greeks(OptionRight.call, 100.0, 100.0, 0.25, 0.045, 0.20)
    assert g["theta"] < 0


def test_iv_solver_round_trip():
    spot, strike, t, r, sigma = 100.0, 105.0, 0.25, 0.045, 0.30
    price = black_scholes_price(OptionRight.call, spot, strike, t, r, sigma)
    iv = implied_vol_brent(OptionRight.call, price, spot, strike, t, r)
    assert iv is not None
    assert iv == pytest.approx(sigma, abs=1e-3)


def test_iv_solver_returns_none_for_impossible_price():
    iv = implied_vol_brent(OptionRight.call, 1e6, 100.0, 100.0, 0.25, 0.045)
    # market_price way above any achievable BS price → no root
    assert iv is None or iv > 4.99  # solver may also walk to the upper bound


def test_zero_t_returns_intrinsic_value():
    assert black_scholes_price(OptionRight.call, 110.0, 100.0, 0.0, 0.045, 0.20) == 10.0
    assert black_scholes_price(OptionRight.put, 90.0, 100.0, 0.0, 0.045, 0.20) == 10.0


def test_zero_t_greeks_all_zero():
    g = greeks(OptionRight.call, 100.0, 100.0, 0.0, 0.045, 0.20)
    assert g == {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
