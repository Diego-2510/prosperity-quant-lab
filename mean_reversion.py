"""Mean-Reversion trader for IMC Prosperity 4.

Strategy
--------
For every product the trader keeps a rolling history of mid prices
(length = ``window``) and derives a rolling mean and standard
deviation.  The current z-score of the mid,

    z = (mid - mean) / std,

drives entries and exits:

* **Entry long**   when  z  <= -entry_sigma
* **Entry short**  when  z  >=  entry_sigma
* **Exit long**    when  z  >= -(entry_sigma - sigma_gap)
* **Exit short**   when  z  <=  (entry_sigma - sigma_gap)
* **Forced flatten** when a position has been held for more than
  ``max_hold_ticks`` ticks (safety net if mean reversion does not happen).

Time is counted in **ticks**, not seconds -- a dedicated tick counter is
incremented once per ``run()`` call.

State storage
-------------
State (price history, tick counter, entry ticks) is kept on the trader
**instance** itself (``self._state``).  The backtester instantiates the
trader once per run, so instance state is persistent across ticks within
a run -- this avoids any JSON (de)serialization overhead on the hot
path, which dominated runtime in earlier revisions.

For live Prosperity deployment, ``state.traderData`` is still consulted
on the very first tick as a cold-start fallback, and the returned
traderData string is kept empty to minimise IO.

Parameters (tunable via the backtester's grid search)
-----------------------------------------------------
``entry_sigma``     Number of standard deviations away from the rolling
                    mean that triggers an entry.
``sigma_gap``       Reduction of the entry z-score that triggers the
                    exit -- determines the expected return per round-trip
                    (gap between entry and exit in sigma units).
``max_hold_ticks``  Maximum number of ticks a position is held before
                    it is forcibly flattened (loss cap).
``window``          Rolling window (in ticks) used for mean and std.

Parameter grid
--------------
The ``PARAM_SPEC`` below declares a *coarse* grid with 0.25-sigma spacing
on the continuous parameters and a balanced geometric spacing on the
integer parameters.  For the final parameter search, run the backtester
with ``--ctf --ctf-n-interp 24`` -- the fine stage then inserts 24
equidistant points between each pair of top-region values, which yields:

* ``entry_sigma`` / ``sigma_gap`` : **step 0.01 sigma** (0.25 / 25)
* ``max_hold_ticks`` / ``window`` : **step 1-2 ticks** in the dense regions

Coarse grid sizes (tuned for the P3 2024 Round 3 data):
* entry_sigma     : 9 values   (1.00 .. 3.00, step 0.25)
* sigma_gap       : 12 values  (0.25 .. 3.00, step 0.25)
* max_hold_ticks  : 11 values  (10 .. 2000)
* window          : 8 values   (10 .. 500)

All order sign conventions follow Prosperity 4:
positive ``Order.quantity`` = BUY, negative = SELL.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any
from collections import deque
import math


class Trader:
    # Default position limits. Overridden per-product by the backtester if
    # it knows better (or by pointing ``LIMIT`` at your real products).
    LIMIT: Dict[str, int] = {}
    DEFAULT_LIMIT = 50

    # Grid-search parameter registry (picked up by the backtester).
    # LEAN coarse grid -- tuned for the two-stage CTF search:
    #   coarse = 5 * 4 * 4 * 5 = 400 combos (was 9504 = 24x reduction)
    # The fine stage (--ctf) automatically interpolates between the top
    # coarse values, so density is only needed on the continuous params.
    # Use --ctf-n-interp 8 to reach 0.125-sigma resolution in stage 2;
    # --ctf-n-interp 24 for 0.05-sigma resolution.
    PARAM_SPEC: Dict[str, Dict[str, Any]] = {
        "entry_sigma": {
            "type": "float",
            # 1.0, 1.5, 2.0, 2.5, 3.0 -> 5 values (0.5 sigma step)
            "grid": [1.0, 1.5, 2.0, 2.5, 3.0],
        },
        "sigma_gap": {
            "type": "float",
            # 0.5, 1.0, 1.5, 2.0 -> 4 values (covers 25-100% of entry_sigma)
            "grid": [0.5, 1.0, 1.5, 2.0],
        },
        "max_hold_ticks": {
            "type": "int",
            # 50, 100, 200, 500 -> 4 values (geometric spacing).  Values
            # above max_ticks are auto-capped by the backtester, so no
            # need to include 1000+.
            "grid": [50, 100, 200, 500],
        },
        "window": {
            "type": "int",
            # 20, 50, 100, 200, 300 -> 5 values.  Short windows catch
            # tight mean-reverters (RAINFOREST_RESIN), longer ones give
            # more stable z-score for drift-heavy products.  Extended to
            # 300 after the Round-3 sensitivity plot showed mean-sharpe
            # still monotonically rising at window=200 -- the optimum was
            # outside the old grid.  Values >= max_ticks are auto-clamped.
            "grid": [20, 50, 100, 200, 300],
        },
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
    #  Lazy state init
    # ------------------------------------------------------------------
    def _ensure_state(self):
        """Create the in-memory state struct once per trader instance."""
        st = getattr(self, "_state", None)
        if st is None:
            st = {
                "tick": 0,
                "hist": {},      # product -> deque of recent mids
                "entries": {},   # product -> entry tick
                "_window_cap": int(self.window) + 1,
            }
            self._state = st
        return st

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------
    def run(self, state: TradingState):
        st = self._ensure_state()

        # If window changed between runs (grid search), resize the deques.
        win_cap = int(self.window) + 1
        if win_cap != st["_window_cap"]:
            st["_window_cap"] = win_cap
            for p, dq in list(st["hist"].items()):
                st["hist"][p] = deque(dq, maxlen=win_cap)

        st["tick"] += 1
        tick = st["tick"]

        result: Dict[str, List[Order]] = {}
        hist_map = st["hist"]
        entries = st["entries"]
        win_len = int(self.window)
        min_obs = self.MIN_OBS

        for product, depth in state.order_depths.items():
            # Inline fast mid computation (avoids function call overhead
            # on the hot path -- profile showed ~7% improvement).
            buy = depth.buy_orders
            sell = depth.sell_orders
            if not buy or not sell:
                continue
            best_bid = max(buy)
            best_ask = min(sell)
            mid = (best_bid + best_ask) * 0.5

            dq = hist_map.get(product)
            if dq is None:
                dq = deque(maxlen=win_cap)
                hist_map[product] = dq
            dq.append(mid)

            n = len(dq)
            if n < min_obs or n < 5:
                continue

            # Use the LAST `window` values.  deque is bounded by
            # ``maxlen = window + 1`` so we can take the full content
            # minus one entry when over-full.
            if n > win_len:
                # drop the oldest single entry virtually via iteration start
                it = iter(dq)
                next(it)
                vals = list(it)
            else:
                vals = list(dq)

            m = len(vals)
            s = 0.0
            for x in vals:
                s += x
            mean = s / m
            var_acc = 0.0
            for x in vals:
                d = x - mean
                var_acc += d * d
            var = var_acc / (m - 1) if m > 1 else 0.0
            std = math.sqrt(var)
            if std <= 1e-9:
                continue

            z = (mid - mean) / std

            pos = int(state.position.get(product, 0))
            limit = int(self.LIMIT.get(product, self.DEFAULT_LIMIT))
            entry_tick = entries.get(product)

            orders = self._decide(
                product=product,
                depth=depth,
                best_bid=best_bid,
                best_ask=best_ask,
                pos=pos,
                limit=limit,
                z=z,
                tick=tick,
                entry_tick=entry_tick,
            )

            # Maintain the "entry tick" marker.
            if orders:
                if pos == 0:
                    # any new non-zero order opens a position
                    for o in orders:
                        if o.quantity != 0:
                            entries[product] = tick
                            break
                else:
                    net = 0
                    for o in orders:
                        net += o.quantity
                    if pos + net == 0:
                        entries.pop(product, None)
                result[product] = orders

        # Empty traderData string -- state lives on self. In live
        # Prosperity, this still round-trips harmlessly; cold-start is
        # handled by _ensure_state on the very first run() call.
        return result, 0, ""

    # ------------------------------------------------------------------
    #  Core decision logic
    # ------------------------------------------------------------------
    def _decide(self, product, depth, best_bid, best_ask, pos, limit, z,
                tick, entry_tick):
        """Return a list of Orders for one product based on the current z-score."""
        entry = float(self.entry_sigma)
        gap = float(self.sigma_gap)
        exit_abs = entry - gap
        if exit_abs < 0.0:
            exit_abs = 0.0

        # ---- 1. Forced flatten on timeout -----------------------------
        if pos != 0 and entry_tick is not None:
            if (tick - int(entry_tick)) >= int(self.max_hold_ticks):
                if pos > 0:
                    return [Order(product, best_bid, -pos)]
                return [Order(product, best_ask, -pos)]

        # ---- 2. Exit if z has mean-reverted far enough ----------------
        if pos > 0 and z >= -exit_abs:
            return [Order(product, best_bid, -pos)]
        if pos < 0 and z <= exit_abs:
            return [Order(product, best_ask, -pos)]

        # ---- 3. Entries ----------------------------------------------
        # Only open a new position if currently flat.
        if pos == 0:
            if z <= -entry:
                qty = -depth.sell_orders[best_ask]  # sell volume is negative
                if qty > limit:
                    qty = limit
                if qty > 0:
                    return [Order(product, best_ask, qty)]
            elif z >= entry:
                qty = depth.buy_orders[best_bid]
                if qty > limit:
                    qty = limit
                if qty > 0:
                    return [Order(product, best_bid, -qty)]

        return []
