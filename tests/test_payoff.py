from __future__ import annotations

import math

import pytest

from optagent.payoff import (
    CONTRACT_MULTIPLIER,
    breakeven,
    max_loss,
    mid_price,
    spread_pct,
)
from optagent.schemas import OptionRight


def test_breakeven_call():
    assert breakeven(OptionRight.call, 100.0, 2.50) == pytest.approx(102.50)


def test_breakeven_put():
    assert breakeven(OptionRight.put, 100.0, 2.50) == pytest.approx(97.50)


def test_breakeven_negative_premium_rejected():
    with pytest.raises(ValueError):
        breakeven(OptionRight.call, 100.0, -0.1)


def test_max_loss_one_contract():
    assert max_loss(2.50) == pytest.approx(250.0)


def test_max_loss_multi_contracts_scales_linearly():
    assert max_loss(2.50, contracts=3) == pytest.approx(750.0)
    assert max_loss(2.50, contracts=3) == 3 * max_loss(2.50)


def test_max_loss_rejects_zero_contracts():
    with pytest.raises(ValueError):
        max_loss(1.0, contracts=0)


def test_contract_multiplier_is_100():
    assert CONTRACT_MULTIPLIER == 100


def test_spread_pct_normal():
    assert spread_pct(1.00, 1.10) == pytest.approx(0.10 / 1.05)


def test_spread_pct_inverted_returns_inf():
    assert spread_pct(1.10, 1.00) == float("inf")


def test_spread_pct_zero_mid_returns_inf():
    assert spread_pct(0.0, 0.0) == float("inf")


def test_mid_price_simple():
    assert mid_price(1.00, 1.10) == pytest.approx(1.05)


def test_mid_price_rejects_negative():
    with pytest.raises(ValueError):
        mid_price(-1.0, 1.0)
