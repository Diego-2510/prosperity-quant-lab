from __future__ import annotations

import math

from scipy.stats import norm


def black_scholes_call(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    rate: float = 0.0,
) -> float:
    """Price a European call option with the Black-Scholes model."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")

    if time_to_expiry < 0:
        raise ValueError("time_to_expiry must be non-negative")

    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    if time_to_expiry == 0 or volatility == 0:
        return max(
            0.0,
            spot - strike * math.exp(-rate * time_to_expiry),
        )

    root_t = math.sqrt(time_to_expiry)

    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * root_t
    )

    d2 = d1 - volatility * root_t

    return float(spot * norm.cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2))


def implied_volatility_call(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float = 0.0,
    *,
    lower: float = 1e-6,
    upper: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """Recover call implied volatility using bounded bisection."""
    if price < 0:
        raise ValueError("option price must be non-negative")

    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")

    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive")

    if lower <= 0 or upper <= lower:
        raise ValueError("volatility bounds must satisfy 0 < lower < upper")

    intrinsic = max(
        0.0,
        spot - strike * math.exp(-rate * time_to_expiry),
    )

    upper_price_bound = spot

    if price < intrinsic - tolerance or price > upper_price_bound + tolerance:
        raise ValueError("option price violates call arbitrage bounds")

    lower_price = black_scholes_call(
        spot,
        strike,
        time_to_expiry,
        lower,
        rate,
    )

    upper_price = black_scholes_call(
        spot,
        strike,
        time_to_expiry,
        upper,
        rate,
    )

    if price < lower_price - tolerance or price > upper_price + tolerance:
        raise ValueError("option price is outside the configured volatility search range")

    lo = lower
    hi = upper

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0

        estimate = black_scholes_call(
            spot,
            strike,
            time_to_expiry,
            mid,
            rate,
        )

        if abs(estimate - price) <= tolerance:
            return mid

        if estimate < price:
            lo = mid
        else:
            hi = mid

    final = (lo + hi) / 2.0

    if (
        abs(
            black_scholes_call(
                spot,
                strike,
                time_to_expiry,
                final,
                rate,
            )
            - price
        )
        <= tolerance * 10
    ):
        return final

    raise ValueError("implied volatility did not converge")
