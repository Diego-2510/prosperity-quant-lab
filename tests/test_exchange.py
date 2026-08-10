from prosperity_quant_lab.exchange import (
    ExchangeError,
    Order,
    OrderBook,
    apply_aggregate_side_limits,
    match_crossing_orders,
)


def book() -> OrderBook:
    return OrderBook(
        buy_orders={99: 5, 98: 5},
        sell_orders={101: -4, 102: -6},
    )


def test_buy_crosses_multiple_ask_levels_at_resting_prices() -> None:
    result = match_crossing_orders(
        {"PEARL": [Order("PEARL", 102, 7)]},
        {"PEARL": book()},
        {"PEARL": 0},
        {"PEARL": 20},
        day=0,
        timestamp=100,
    )

    assert [(fill.price, fill.quantity) for fill in result.fills] == [(101, 4), (102, 3)]
    assert result.position_delta == {"PEARL": 7}


def test_sell_crosses_multiple_bid_levels_at_resting_prices() -> None:
    result = match_crossing_orders(
        {"PEARL": [Order("PEARL", 98, -7)]},
        {"PEARL": book()},
        {"PEARL": 0},
        {"PEARL": 20},
        day=0,
        timestamp=100,
    )

    assert [(fill.price, fill.quantity) for fill in result.fills] == [(99, -5), (98, -2)]
    assert result.position_delta == {"PEARL": -7}


def test_aggregate_buy_side_is_rejected_when_submitted_side_breaches_limit() -> None:
    kept, rejected_buys, rejected_sells = apply_aggregate_side_limits(
        [Order("PEARL", 101, 2), Order("PEARL", 102, 1), Order("PEARL", 99, -4)],
        current_position=8,
        position_limit=10,
    )

    assert kept == [Order("PEARL", 99, -4)]
    assert rejected_buys
    assert not rejected_sells


def test_aggregate_sell_side_is_rejected_when_submitted_side_breaches_limit() -> None:
    kept, rejected_buys, rejected_sells = apply_aggregate_side_limits(
        [Order("PEARL", 99, -2), Order("PEARL", 98, -1), Order("PEARL", 101, 2)],
        current_position=-8,
        position_limit=10,
    )

    assert kept == [Order("PEARL", 101, 2)]
    assert not rejected_buys
    assert rejected_sells


def test_non_marketable_order_does_not_receive_passive_fill() -> None:
    result = match_crossing_orders(
        {"PEARL": [Order("PEARL", 100, 3)]},
        {"PEARL": book()},
        {"PEARL": 0},
        {"PEARL": 20},
        day=0,
        timestamp=100,
    )

    assert result.fills == ()
    assert result.position_delta == {}


def test_missing_position_limit_is_an_error() -> None:
    try:
        match_crossing_orders(
            {"PEARL": [Order("PEARL", 102, 1)]},
            {"PEARL": book()},
            {},
            {},
            day=0,
            timestamp=100,
        )
    except ExchangeError as exc:
        assert "missing position limit" in str(exc)
    else:
        raise AssertionError("expected ExchangeError")
