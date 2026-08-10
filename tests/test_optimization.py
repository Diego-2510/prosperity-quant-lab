import pandas as pd

from prosperity_quant_lab.optimization import expand_grid, grid_search
from prosperity_quant_lab.strategies import MeanReversionStrategy


def test_expand_grid_is_deterministic() -> None:
    assert expand_grid({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_grid_search_returns_one_row_per_configuration() -> None:
    prices = pd.DataFrame(
        [
            {
                "day": 0,
                "timestamp": timestamp,
                "product": "A",
                "bid_price_1": 99 + i,
                "bid_volume_1": 10,
                "ask_price_1": 101 + i,
                "ask_volume_1": 10,
                "mid_price": 100 + i,
            }
            for i, timestamp in enumerate([0, 100, 200, 300, 400])
        ]
    )

    result = grid_search(
        prices,
        lambda params: MeanReversionStrategy(window=params["window"], z_entry=10.0, order_size=1),
        {"window": [2, 3]},
        {"A": 10},
    )

    assert len(result) == 2
    assert set(result["window"]) == {2, 3}
