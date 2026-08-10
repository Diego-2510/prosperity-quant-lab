import numpy as np
import pandas as pd

from prosperity_quant_lab.exchange import Order
from prosperity_quant_lab.simulation import moving_block_bootstrap, replay


class BuyOnce:
    def __init__(self) -> None:
        self.done = False

    def on_tick(self, state):
        if self.done:
            return {}
        self.done = True
        return {"A": [Order("A", state.books["A"].best_ask, 2)]}


def prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "day": 0,
                "timestamp": 0,
                "product": "A",
                "bid_price_1": 99,
                "bid_volume_1": 5,
                "ask_price_1": 101,
                "ask_volume_1": 5,
                "mid_price": 100,
            },
            {
                "day": 0,
                "timestamp": 100,
                "product": "A",
                "bid_price_1": 101,
                "bid_volume_1": 5,
                "ask_price_1": 103,
                "ask_volume_1": 5,
                "mid_price": 102,
            },
        ]
    )


def test_replay_marks_positions_to_historical_mid() -> None:
    result = replay(prices(), BuyOnce(), {"A": 10})

    assert result.final_positions == {"A": 2}
    assert result.final_cash == {"A": -202.0}
    assert result.metrics["final_pnl"] == 2.0
    assert result.metrics["fill_count"] == 1


def test_moving_block_bootstrap_is_seed_reproducible() -> None:
    returns = np.arange(10, dtype=float)
    first = moving_block_bootstrap(returns, path_length=12, block_length=3, seed=42)
    second = moving_block_bootstrap(returns, path_length=12, block_length=3, seed=42)

    assert np.array_equal(first, second)
    assert len(first) == 12
