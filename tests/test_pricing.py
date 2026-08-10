import pytest

from prosperity_quant_lab.pricing import black_scholes_call, implied_volatility_call


def test_black_scholes_at_expiry_is_intrinsic_value() -> None:
    assert black_scholes_call(110, 100, 0.0, 0.2) == 10.0


def test_implied_volatility_recovers_known_volatility() -> None:
    price = black_scholes_call(100, 100, 1.0, 0.25)
    recovered = implied_volatility_call(price, 100, 100, 1.0)
    assert recovered == pytest.approx(0.25, abs=1e-6)


def test_implied_volatility_rejects_arbitrage_violating_price() -> None:
    with pytest.raises(ValueError, match="arbitrage bounds"):
        implied_volatility_call(150, 100, 100, 1.0)
