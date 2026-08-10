"""
IMC Prosperity 4 — Round 4 algorithmic trader (multi-strike version).

Strategy: directional mean-reversion on the two underlyings (HYDROGEL_PACK,
VELVETFRUIT_EXTRACT). The VELVETFRUIT_EXTRACT z-score also drives a basket of
ITM/ATM call options, each used as a leveraged proxy on the same signal.

Backtested on round-4 days 1+2+3:
  Single strike (VEV_5000 @300):                +363,893
  Two strikes  (VEV_5000+VEV_5100 @300 each):   +498,210
  Three strikes (VEV_5000+5100+5200 @300):      +602,598   ← BEST risk/reward
  Five strikes  (4500..5300 @300):              +742,037   (more PnL but worse PnL/risk)

The 3-strike ATM stack (5000+5100+5200) maximizes PnL/delta-exposure (~664)
and PnL/maxDD (~9.3). Going wider adds PnL but at deteriorating risk-adjusted
return because OTM strikes (5300+) have low effective delta (R²<0.4) and ITM
strikes (4500-) have a wider bid-ask that eats the edge.

Persistent state (carried in traderData as JSON):
  - target_pos: sticky targets per symbol
  - tick_count
  - history: rolling mid-price buffer per underlying (online-calibration fallback)
"""

import json
import math

from datamodel import Order, OrderDepth, TradingState

# =============================================================================
# CONFIGURATION
# =============================================================================

UNDERLYINGS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]

# === MULTI-STRIKE OPTION BASKET ===
# Each entry: symbol -> position limit (cap per strike).
# Recommended config: 3 ATM strikes for best PnL/risk.
# To trade only one strike (original strategy), set OPTION_LIMITS = {"VEV_5000": 300}.
OPTION_LIMITS: dict[str, int] = {
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    # "VEV_4500": 300,  # uncomment to add — gives +PnL but lower R²
    # "VEV_5300": 300,  # uncomment to add — gives +PnL but smaller delta per unit risk
}

# Position limits for underlyings
UNDERLYING_LIMIT = 200

POSITION_LIMIT: dict[str, int] = {
    "HYDROGEL_PACK": UNDERLYING_LIMIT,
    "VELVETFRUIT_EXTRACT": UNDERLYING_LIMIT,
    **OPTION_LIMITS,
}

ALL_PRODUCTS = UNDERLYINGS + list(OPTION_LIMITS.keys())

# Z-score entry threshold (sweep optimum: 1.0)
Z_ENTRY = 1.0

# Hardcoded calibration (round-4 days 1+2 mid-prices)
HARDCODED_PARAMS: dict[str, dict[str, float]] = {
    "HYDROGEL_PACK": {"mean": 9990.73, "std": 34.77},
    "VELVETFRUIT_EXTRACT": {"mean": 5251.89, "std": 16.23},
}

# Optional online calibration on rolling window (turn on if you suspect drift)
USE_ONLINE_CALIBRATION = False
ROLLING_WINDOW = 2000
ONLINE_RECOMPUTE_EVERY = 100

# Walk-the-book max levels (Prosperity exposes up to 3)
MAX_LEVELS = 3


# =============================================================================
# Helpers
# =============================================================================


def best_bid_ask_mid(depth: OrderDepth):
    bids = sorted(depth.buy_orders.keys(), reverse=True) if depth.buy_orders else []
    asks = sorted(depth.sell_orders.keys()) if depth.sell_orders else []
    if not bids or not asks:
        return None, None, None
    return bids[0], asks[0], 0.5 * (bids[0] + asks[0])


def walk_book_orders(symbol: str, qty: int, depth: OrderDepth) -> list[Order]:
    """Build orders that consume up to MAX_LEVELS levels to execute `qty` (signed)."""
    orders: list[Order] = []
    if qty == 0:
        return orders
    if qty > 0:
        # Buy from sell_orders (volumes are negative in IMC convention)
        levels = sorted(depth.sell_orders.items())
        remaining = qty
        for price, vol in levels[:MAX_LEVELS]:
            available = -vol
            if available <= 0:
                continue
            take = min(remaining, available)
            if take <= 0:
                continue
            orders.append(Order(symbol, price, take))
            remaining -= take
            if remaining == 0:
                break
    else:
        levels = sorted(depth.buy_orders.items(), reverse=True)
        remaining = -qty
        for price, vol in levels[:MAX_LEVELS]:
            available = vol
            if available <= 0:
                continue
            take = min(remaining, available)
            if take <= 0:
                continue
            orders.append(Order(symbol, price, -take))
            remaining -= take
            if remaining == 0:
                break
    return orders


def clamp_to_limit(symbol: str, current_pos: int, target: int) -> int:
    limit = POSITION_LIMIT[symbol]
    target = max(-limit, min(limit, target))
    return target - current_pos


def compute_zscore(mid: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 0.0
    return (mid - mu) / sigma


def online_mu_sigma(history: list[float]):
    n = len(history)
    if n < 2:
        return 0.0, 0.0
    mu = sum(history) / n
    var = sum((x - mu) ** 2 for x in history) / n
    return mu, math.sqrt(var)


# =============================================================================
# Trader
# =============================================================================


class Trader:
    def run(self, state: TradingState):
        # ---------- Load persistent state ----------
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}

        target_pos: dict[str, int] = mem.get("target_pos", {p: 0 for p in ALL_PRODUCTS})
        for p in ALL_PRODUCTS:
            target_pos.setdefault(p, 0)

        history: dict[str, list[float]] = mem.get("history", {p: [] for p in UNDERLYINGS})
        for p in UNDERLYINGS:
            history.setdefault(p, [])

        tick_count: int = mem.get("tick_count", 0) + 1

        # ---------- Determine current parameters ----------
        params = dict(HARDCODED_PARAMS)
        if USE_ONLINE_CALIBRATION and tick_count % ONLINE_RECOMPUTE_EVERY == 0:
            for p in UNDERLYINGS:
                if len(history[p]) >= ROLLING_WINDOW // 4:
                    mu, sd = online_mu_sigma(history[p][-ROLLING_WINDOW:])
                    if sd > 1e-6:
                        params[p] = {"mean": mu, "std": sd}

        # ---------- Build orders ----------
        result: dict[str, list[Order]] = {}

        # --- Underlyings ---
        velvet_z = 0.0
        for product in UNDERLYINGS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            _, _, mid = best_bid_ask_mid(depth)
            if mid is None:
                continue

            # Update rolling history
            history[product].append(mid)
            if len(history[product]) > ROLLING_WINDOW * 2:
                history[product] = history[product][-ROLLING_WINDOW:]

            mu = params[product]["mean"]
            sd = params[product]["std"]
            z = compute_zscore(mid, mu, sd)
            if product == "VELVETFRUIT_EXTRACT":
                velvet_z = z

            limit = POSITION_LIMIT[product]
            if z > Z_ENTRY:
                target_pos[product] = -limit
            elif z < -Z_ENTRY:
                target_pos[product] = limit
            # else: keep previous (sticky)

            current = state.position.get(product, 0)
            qty = clamp_to_limit(product, current, target_pos[product])
            if qty != 0:
                ords = walk_book_orders(product, qty, depth)
                if ords:
                    result[product] = ords

        # --- Option basket: all driven by VELVET z-score ---
        for option_sym, opt_limit in OPTION_LIMITS.items():
            depth_opt = state.order_depths.get(option_sym)
            if depth_opt is None:
                continue
            _, _, mid_opt = best_bid_ask_mid(depth_opt)
            if mid_opt is None:
                continue

            if velvet_z > Z_ENTRY:
                target_pos[option_sym] = -opt_limit
            elif velvet_z < -Z_ENTRY:
                target_pos[option_sym] = opt_limit
            # else: keep previous (sticky)

            current = state.position.get(option_sym, 0)
            qty = clamp_to_limit(option_sym, current, target_pos[option_sym])
            if qty != 0:
                ords = walk_book_orders(option_sym, qty, depth_opt)
                if ords:
                    result[option_sym] = ords

        # ---------- Persist memory ----------
        new_mem = {
            "target_pos": target_pos,
            "history": {p: history[p][-ROLLING_WINDOW:] for p in UNDERLYINGS},
            "tick_count": tick_count,
        }
        try:
            traderData = json.dumps(new_mem, separators=(",", ":"))
        except Exception:
            traderData = ""

        conversions = 0
        return result, conversions, traderData
