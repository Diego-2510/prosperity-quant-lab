"""Mean-Reversion trader for IMC Prosperity 4.

Strategy
--------
For every product the trader keeps a rolling history of mid prices
(length = ``window``) inside ``traderData`` and derives a rolling mean
and standard deviation.  The current z-score of the mid,

    z = (mid - mean) / std,

drives entries and exits:

* **Entry long**   when  z  <= -entry_sigma
* **Entry short**  when  z  >=  entry_sigma
* **Exit long**    when  z  >= -(entry_sigma - sigma_gap)
* **Exit short**   when  z  <=  (entry_sigma - sigma_gap)
* **Forced flatten** when a position has been held for more than
  ``max_hold_ticks`` ticks (safety net if mean reversion does not happen).

Time is counted in **ticks**, not seconds -- a dedicated tick counter is
incremented once per ``run()`` call and stored in ``traderData``.

Parameters (tunable via the backtester's grid search)
-----------------------------------------------------
``entry_sigma``     Number of standard deviations away from the rolling
                    mean that triggers an entry.
``sigma_gap``       Reduction of the entry z-score that triggers the
                    exit -- determines the expected return per trade
                    (gap between entry and exit in sigma units).
``max_hold_ticks``  Maximum number of ticks a position is held before
                    it is forcibly flattened.
``window``          Rolling window size used for mean and std.

All order sign conventions follow Prosperity 4:
positive ``Order.quantity`` = BUY, negative = SELL.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any
import json
import math


class Trader:
    # Default position limits. Overridden per-product by the backtester if
    # it knows better (or by pointing ``LIMIT`` at your real products).
    LIMIT: Dict[str, int] = {}
    DEFAULT_LIMIT = 50

    # Grid-search parameter registry (picked up by the backtester).
    PARAM_SPEC: Dict[str, Dict[str, Any]] = {
        "entry_sigma":    {"type": "float", "grid": [1.5, 2.0, 2.5, 3.0]},
        "sigma_gap":      {"type": "float", "grid": [0.5, 1.0, 1.5, 2.0]},
        "max_hold_ticks": {"type": "int",   "grid": [50, 100, 200, 400]},
        "window":         {"type": "int",   "grid": [50, 100, 200]},
    }

    # Instance-level defaults (overwritten by the backtester via setattr
    # when a grid combination is evaluated).
    entry_sigma: float = 2.0
    sigma_gap: float = 1.0
    max_hold_ticks: int = 200
    window: int = 100

    # Minimum number of observations before the first trade is allowed.
    MIN_OBS = 20

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------
    def run(self, state: TradingState):
        td = self._load_td(state.traderData)
        tick = int(td.get("tick", 0)) + 1
        td["tick"] = tick

        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is None:
                continue

            hist = td.setdefault("hist", {}).setdefault(product, [])
            hist.append(mid)
            if len(hist) > int(self.window) + 1:
                # keep the buffer bounded
                del hist[: len(hist) - int(self.window) - 1]

            if len(hist) < max(self.MIN_OBS, 5):
                continue

            # Use the LAST `window` values (exclusive of the current mid is
            # also valid; we include the current mid so the z-score reflects
            # the latest observation).
            win = hist[-int(self.window):]
            mean = sum(win) / len(win)
            var = sum((x - mean) ** 2 for x in win) / max(1, len(win) - 1)
            std = math.sqrt(var)
            if std <= 1e-9:
                continue

            z = (mid - mean) / std

            pos = int(state.position.get(product, 0))
            limit = int(self.LIMIT.get(product, self.DEFAULT_LIMIT))

            entries = td.setdefault("entries", {})
            entry_tick = entries.get(product)  # tick at which current pos was opened

            orders = self._decide(
                product=product,
                depth=depth,
                pos=pos,
                limit=limit,
                z=z,
                tick=tick,
                entry_tick=entry_tick,
            )

            # Maintain the "entry tick" marker.
            if pos == 0 and any(o.quantity != 0 for o in orders):
                entries[product] = tick
            elif pos != 0 and self._flattens(orders, pos):
                entries.pop(product, None)

            if orders:
                result[product] = orders

        return result, 0, self._dump_td(td)

    # ------------------------------------------------------------------
    #  Core decision logic
    # ------------------------------------------------------------------
    def _decide(self, product, depth, pos, limit, z, tick, entry_tick):
        """Return a list of Orders for one product based on the current z-score."""
        orders: List[Order] = []

        best_bid = max(depth.buy_orders.keys())  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders

        entry = float(self.entry_sigma)
        gap   = float(self.sigma_gap)
        # Exit trigger is `entry - gap` sigma on the opposite side of zero.
        # Example: entry 2.0, gap 1.0 -> exit at |z| <= 1.0.
        exit_abs = max(0.0, entry - gap)

        # ---- 1. Forced flatten on timeout -----------------------------
        if pos != 0 and entry_tick is not None:
            held = tick - int(entry_tick)
            if held >= int(self.max_hold_ticks):
                return self._flatten(product, pos, best_bid, best_ask)

        # ---- 2. Exit if z has mean-reverted far enough ----------------
        if pos > 0 and z >= -exit_abs:
            return self._flatten(product, pos, best_bid, best_ask)
        if pos < 0 and z <=  exit_abs:
            return self._flatten(product, pos, best_bid, best_ask)

        # ---- 3. Entries ----------------------------------------------
        # Only open a new position if we are currently flat, to keep the
        # strategy interpretable (one round-trip at a time per product).
        if pos == 0:
            if z <= -entry:
                # Long entry -- hit the ask.
                qty = min(abs(depth.sell_orders[best_ask]), limit)
                if qty > 0:
                    orders.append(Order(product, best_ask,  qty))
            elif z >= entry:
                # Short entry -- hit the bid.
                qty = min(depth.buy_orders[best_bid], limit)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

        return orders

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mid(depth: OrderDepth):
        bb = max(depth.buy_orders.keys())  if depth.buy_orders  else None
        ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @staticmethod
    def _flatten(product, pos, best_bid, best_ask):
        """Aggressive flatten: cross the spread for the full position."""
        if pos > 0:
            return [Order(product, best_bid, -pos)]
        if pos < 0:
            return [Order(product, best_ask, -pos)]  # -pos > 0 -> BUY
        return []

    @staticmethod
    def _flattens(orders: List[Order], pos: int) -> bool:
        """True if the submitted orders fully close the current position."""
        net = sum(o.quantity for o in orders)
        return pos + net == 0

    @staticmethod
    def _load_td(s: str) -> Dict[str, Any]:
        if not s:
            return {}
        try:
            d = json.loads(s)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _dump_td(td: Dict[str, Any]) -> str:
        try:
            return json.dumps(td, separators=(",", ":"))
        except Exception:
            return "{}"
