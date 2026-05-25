"""Black-Scholes Greeks and IV sanity-check helpers.

Used ONLY for analytics on contracts the screener already produced from
broker data. NEVER used to compute breakeven/max_loss — those live in
`payoff.py` and are payoff math, not pricing-model outputs.
"""

from __future__ import annotations

import math

from .schemas import OptionRight


SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def _phi(x: float) -> float:
    """Standard-normal PDF."""

    return math.exp(-0.5 * x * x) / SQRT_TWO_PI


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(spot: float, strike: float, t_years: float, r: float, q: float, sigma: float) -> float:
    return (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t_years) / (
        sigma * math.sqrt(t_years)
    )


def _d2(d1: float, sigma: float, t_years: float) -> float:
    return d1 - sigma * math.sqrt(t_years)


def black_scholes_price(
    right: OptionRight,
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """European Black-Scholes price (continuous dividend yield q).

    Used only as an independent check against broker mid; not the source of
    truth for the screener output.
    """

    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if t_years <= 0 or sigma <= 0:
        # at expiration or zero vol: intrinsic value
        if right is OptionRight.call:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    d1 = _d1(spot, strike, t_years, r, q, sigma)
    d2 = _d2(d1, sigma, t_years)
    disc_r = math.exp(-r * t_years)
    disc_q = math.exp(-q * t_years)
    if right is OptionRight.call:
        return spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
    return strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)


def greeks(
    right: OptionRight,
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> dict[str, float]:
    """Delta, gamma, vega (per 1.00 vol move), theta (per day), rho.

    Returns a dict; callers can pluck whichever Greeks they need. Theta is
    reported per calendar day (annual theta / 365) for screener readability.
    """

    if t_years <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = _d1(spot, strike, t_years, r, q, sigma)
    d2 = _d2(d1, sigma, t_years)
    disc_r = math.exp(-r * t_years)
    disc_q = math.exp(-q * t_years)
    sqrt_t = math.sqrt(t_years)

    if right is OptionRight.call:
        delta = disc_q * _norm_cdf(d1)
        theta_annual = (
            -(spot * disc_q * _phi(d1) * sigma) / (2.0 * sqrt_t)
            - r * strike * disc_r * _norm_cdf(d2)
            + q * spot * disc_q * _norm_cdf(d1)
        )
        rho = strike * t_years * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta_annual = (
            -(spot * disc_q * _phi(d1) * sigma) / (2.0 * sqrt_t)
            + r * strike * disc_r * _norm_cdf(-d2)
            - q * spot * disc_q * _norm_cdf(-d1)
        )
        rho = -strike * t_years * disc_r * _norm_cdf(-d2)

    gamma = disc_q * _phi(d1) / (spot * sigma * sqrt_t)
    vega = spot * disc_q * _phi(d1) * sqrt_t  # per 1.00 vol move

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_annual / 365.0,
        "rho": rho,
    }


def implied_vol_brent(
    right: OptionRight,
    market_price: float,
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    q: float = 0.0,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Bisection IV solver. Returns None when the market price is outside the
    achievable BS range (no positive sigma reproduces the price).
    """

    if market_price <= 0 or spot <= 0 or strike <= 0 or t_years <= 0:
        return None

    def f(sigma: float) -> float:
        return black_scholes_price(right, spot, strike, t_years, r, sigma, q) - market_price

    lo, hi = sigma_lo, sigma_hi
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_mid * f_lo < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
