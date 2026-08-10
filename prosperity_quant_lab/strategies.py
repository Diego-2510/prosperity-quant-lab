from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol

import numpy as np

from prosperity_quant_lab.exchange import (
    MarketState,
    Order,
)


class Strategy(Protocol):
    def on_tick(
        self,
        state: MarketState,
    ) -> dict[
        str,
        list[Order],
    ]: ...


class MeanReversionStrategy:
    """Simple reproducible research example."""

    def __init__(
        self,
        *,
        window: int = 20,
        z_entry: float = 2.0,
        order_size: int = 5,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")

        if z_entry <= 0:
            raise ValueError("z_entry must be positive")

        if order_size <= 0:
            raise ValueError("order_size must be positive")

        self.window = int(window)

        self.z_entry = float(z_entry)

        self.order_size = int(order_size)

        self._history: defaultdict[
            str,
            deque[float],
        ] = defaultdict(lambda: deque(maxlen=self.window))

    def on_tick(
        self,
        state: MarketState,
    ) -> dict[
        str,
        list[Order],
    ]:
        output: dict[
            str,
            list[Order],
        ] = {}

        for (
            product,
            mid,
        ) in state.mids.items():
            history = self._history[product]

            history.append(float(mid))

            if len(history) < self.window:
                continue

            values = np.asarray(
                history,
                dtype=float,
            )

            std = float(values.std(ddof=1))

            if std <= 1e-12:
                continue

            zscore = (float(mid) - float(values.mean())) / std

            book = state.books[product]

            if zscore <= -self.z_entry and book.best_ask is not None:
                output[product] = [
                    Order(
                        product,
                        book.best_ask,
                        self.order_size,
                    )
                ]

            elif zscore >= self.z_entry and book.best_bid is not None:
                output[product] = [
                    Order(
                        product,
                        book.best_bid,
                        -self.order_size,
                    )
                ]

        return output
