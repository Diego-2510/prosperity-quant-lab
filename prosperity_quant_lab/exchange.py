from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


class ExchangeError(ValueError):
    """Raised when a replay request violates the exchange model contract."""


@dataclass(frozen=True)
class Order:
    product: str
    price: int
    quantity: int

    def __post_init__(self) -> None:
        if not self.product:
            raise ExchangeError("order product must be non-empty")
        if self.price <= 0:
            raise ExchangeError("order price must be positive")
        if self.quantity == 0:
            raise ExchangeError("order quantity must be non-zero")


@dataclass
class OrderBook:
    buy_orders: dict[int, int] = field(default_factory=dict)
    sell_orders: dict[int, int] = field(default_factory=dict)

    def validate(self) -> None:
        if any(price <= 0 or volume <= 0 for price, volume in self.buy_orders.items()):
            raise ExchangeError("buy book must contain positive prices and volumes")

        if any(price <= 0 or volume >= 0 for price, volume in self.sell_orders.items()):
            raise ExchangeError("sell book must contain positive prices and negative volumes")

        if self.buy_orders and self.sell_orders:
            best_bid = max(self.buy_orders)
            best_ask = min(self.sell_orders)

            if best_bid >= best_ask:
                raise ExchangeError(
                    f"crossed or locked historical book: best_bid={best_bid}, best_ask={best_ask}"
                )

    @property
    def best_bid(self) -> int | None:
        return max(self.buy_orders) if self.buy_orders else None

    @property
    def best_ask(self) -> int | None:
        return min(self.sell_orders) if self.sell_orders else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None

        return (self.best_bid + self.best_ask) / 2.0


@dataclass(frozen=True)
class Fill:
    product: str
    price: int
    quantity: int
    day: int
    timestamp: int


@dataclass(frozen=True)
class MarketState:
    day: int
    timestamp: int
    books: Mapping[str, OrderBook]
    positions: Mapping[str, int]
    mids: Mapping[str, float]


@dataclass(frozen=True)
class MatchResult:
    fills: tuple[Fill, ...]
    position_delta: Mapping[str, int]
    rejected_buy_products: frozenset[str]
    rejected_sell_products: frozenset[str]


def _validate_limit(
    current_position: int,
    position_limit: int,
) -> None:
    if position_limit <= 0:
        raise ExchangeError("position limit must be positive")

    if abs(current_position) > position_limit:
        raise ExchangeError(
            f"current position {current_position} already exceeds limit {position_limit}"
        )


def apply_aggregate_side_limits(
    orders: Sequence[Order],
    current_position: int,
    position_limit: int,
) -> tuple[
    list[Order],
    bool,
    bool,
]:
    """Apply the documented side-aggregate position-limit rule."""
    _validate_limit(
        current_position,
        position_limit,
    )

    buys = [order for order in orders if order.quantity > 0]

    sells = [order for order in orders if order.quantity < 0]

    buy_total = sum(order.quantity for order in buys)

    sell_total = sum(order.quantity for order in sells)

    reject_buys = bool(buys and current_position + buy_total > position_limit)

    reject_sells = bool(sells and current_position + sell_total < -position_limit)

    kept: list[Order] = []

    if not reject_buys:
        kept.extend(buys)

    if not reject_sells:
        kept.extend(sells)

    return (
        kept,
        reject_buys,
        reject_sells,
    )


def match_crossing_orders(
    orders_by_product: Mapping[
        str,
        Sequence[Order],
    ],
    books: Mapping[
        str,
        OrderBook,
    ],
    positions: Mapping[
        str,
        int,
    ],
    position_limits: Mapping[
        str,
        int,
    ],
    *,
    day: int,
    timestamp: int,
) -> MatchResult:
    """Match immediately marketable orders against historical books.

    This deliberately does not model queue position, passive fills,
    bot flow, conversions, or order persistence beyond the tick.
    """
    fills: list[Fill] = []

    delta: defaultdict[
        str,
        int,
    ] = defaultdict(int)

    rejected_buys: set[str] = set()
    rejected_sells: set[str] = set()

    for (
        product,
        submitted_orders,
    ) in orders_by_product.items():
        orders = list(submitted_orders)

        if not orders:
            continue

        if any(order.product != product for order in orders):
            raise ExchangeError(f"order product does not match mapping key {product}")

        if product not in position_limits:
            raise ExchangeError(f"missing position limit for product {product}")

        book = books.get(product)

        if book is None:
            continue

        book.validate()

        current_position = int(
            positions.get(
                product,
                0,
            )
        )

        limit = int(position_limits[product])

        (
            kept,
            reject_buy_side,
            reject_sell_side,
        ) = apply_aggregate_side_limits(
            orders,
            current_position,
            limit,
        )

        if reject_buy_side:
            rejected_buys.add(product)

        if reject_sell_side:
            rejected_sells.add(product)

        if not kept:
            continue

        sell_book = dict(book.sell_orders)

        buy_book = dict(book.buy_orders)

        working_position = current_position

        kept.sort(key=lambda order: -order.price if order.quantity > 0 else order.price)

        for order in kept:
            remaining = order.quantity

            if remaining > 0:
                for ask_price in sorted(sell_book):
                    if remaining <= 0 or ask_price > order.price:
                        break

                    available = -sell_book[ask_price]

                    limit_capacity = max(
                        0,
                        limit - working_position,
                    )

                    take = min(
                        remaining,
                        available,
                        limit_capacity,
                    )

                    if take <= 0:
                        break

                    fills.append(
                        Fill(
                            product,
                            ask_price,
                            take,
                            day,
                            timestamp,
                        )
                    )

                    delta[product] += take

                    working_position += take

                    remaining -= take
                    available -= take

                    if available == 0:
                        del sell_book[ask_price]

                    else:
                        sell_book[ask_price] = -available

            else:
                for bid_price in sorted(
                    buy_book,
                    reverse=True,
                ):
                    if remaining >= 0 or bid_price < order.price:
                        break

                    available = buy_book[bid_price]

                    limit_capacity = max(
                        0,
                        limit + working_position,
                    )

                    take = min(
                        -remaining,
                        available,
                        limit_capacity,
                    )

                    if take <= 0:
                        break

                    fills.append(
                        Fill(
                            product,
                            bid_price,
                            -take,
                            day,
                            timestamp,
                        )
                    )

                    delta[product] -= take

                    working_position -= take

                    remaining += take
                    available -= take

                    if available == 0:
                        del buy_book[bid_price]

                    else:
                        buy_book[bid_price] = available

    return MatchResult(
        fills=tuple(fills),
        position_delta=dict(delta),
        rejected_buy_products=frozenset(rejected_buys),
        rejected_sell_products=frozenset(rejected_sells),
    )
