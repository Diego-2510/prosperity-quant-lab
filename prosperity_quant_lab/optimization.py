from __future__ import annotations

import itertools
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from typing import Any

import pandas as pd

from prosperity_quant_lab.simulation import (
    replay,
)
from prosperity_quant_lab.strategies import (
    Strategy,
)


def expand_grid(
    grid: Mapping[
        str,
        Sequence[Any],
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    if not grid:
        return [{}]

    keys = list(grid)

    if any(len(grid[key]) == 0 for key in keys):
        raise ValueError("parameter grid values must be non-empty")

    combinations = itertools.product(*(grid[key] for key in keys))

    return [
        dict(
            zip(
                keys,
                values,
                strict=True,
            )
        )
        for values in combinations
    ]


def grid_search(
    prices: pd.DataFrame,
    strategy_factory: Callable[
        [
            dict[
                str,
                Any,
            ]
        ],
        Strategy,
    ],
    parameter_grid: Mapping[
        str,
        Sequence[Any],
    ],
    position_limits: Mapping[
        str,
        int,
    ],
) -> pd.DataFrame:
    records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for params in expand_grid(parameter_grid):
        result = replay(
            prices,
            strategy_factory(params),
            position_limits,
        )

        records.append(
            {
                **params,
                **result.metrics,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "final_pnl",
                "max_drawdown",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
