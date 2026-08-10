from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prosperity_quant_lab.data import (
    snapshots_from_prices,
)
from prosperity_quant_lab.exchange import (
    Fill,
    MarketState,
    match_crossing_orders,
)
from prosperity_quant_lab.metrics import (
    compute_metrics,
)
from prosperity_quant_lab.strategies import (
    Strategy,
)


@dataclass(frozen=True)
class ReplayResult:
    fills: pd.DataFrame
    equity_curve: pd.DataFrame
    final_positions: Mapping[
        str,
        int,
    ]
    final_cash: Mapping[
        str,
        float,
    ]
    metrics: Mapping[
        str,
        float | int,
    ]


def replay(
    prices: pd.DataFrame,
    strategy: Strategy,
    position_limits: Mapping[
        str,
        int,
    ],
) -> ReplayResult:
    snapshots = snapshots_from_prices(prices)

    if not snapshots:
        raise ValueError("no snapshots available for replay")

    positions: defaultdict[
        str,
        int,
    ] = defaultdict(int)

    cash: defaultdict[
        str,
        float,
    ] = defaultdict(float)

    fills: list[Fill] = []

    equity_rows: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    last_mids: dict[
        str,
        float,
    ] = {}

    for snapshot in snapshots:
        last_mids.update(snapshot.mids)

        state = MarketState(
            day=snapshot.day,
            timestamp=snapshot.timestamp,
            books=snapshot.books,
            positions=dict(positions),
            mids=snapshot.mids,
        )

        orders = strategy.on_tick(state)

        result = match_crossing_orders(
            orders,
            snapshot.books,
            positions,
            position_limits,
            day=snapshot.day,
            timestamp=(snapshot.timestamp),
        )

        for fill in result.fills:
            fills.append(fill)

            cash[fill.product] -= fill.price * fill.quantity

        for (
            product,
            delta,
        ) in result.position_delta.items():
            positions[product] += int(delta)

        marked_equity = float(sum(cash.values()))

        for (
            product,
            position,
        ) in positions.items():
            mid = last_mids.get(product)

            if mid is not None:
                marked_equity += position * mid

        equity_rows.append(
            {
                "day": snapshot.day,
                "timestamp": (snapshot.timestamp),
                "equity": float(marked_equity),
            }
        )

    fill_frame = pd.DataFrame(
        [
            {
                "product": (fill.product),
                "price": fill.price,
                "quantity": (fill.quantity),
                "day": fill.day,
                "timestamp": (fill.timestamp),
            }
            for fill in fills
        ],
        columns=[
            "product",
            "price",
            "quantity",
            "day",
            "timestamp",
        ],
    )

    equity_frame = pd.DataFrame(equity_rows)

    metrics = compute_metrics(
        equity_frame,
        fill_frame,
    )

    return ReplayResult(
        fills=fill_frame,
        equity_curve=equity_frame,
        final_positions=dict(positions),
        final_cash=dict(cash),
        metrics=metrics,
    )


def moving_block_bootstrap(
    returns: np.ndarray,
    *,
    path_length: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Generate one seeded moving-block-bootstrap path."""
    values = np.asarray(
        returns,
        dtype=float,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("returns must be a non-empty one-dimensional array")

    if path_length <= 0:
        raise ValueError("path_length must be positive")

    if block_length <= 0 or block_length > len(values):
        raise ValueError("block_length must be in [1, len(returns)]")

    rng = np.random.default_rng(seed)

    max_start = len(values) - block_length

    output: list[float] = []

    while len(output) < path_length:
        start = int(
            rng.integers(
                0,
                max_start + 1,
            )
        )

        output.extend(values[start : start + block_length].tolist())

    return np.asarray(
        output[:path_length],
        dtype=float,
    )
