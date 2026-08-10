"""STRATEGIE FINALE ROUND 4 — Mean reversion pyramidée multi-produits.

Basée sur la stratégie d'Alexandre (R3 +213k SeaShells), avec :
  1. Position limits CORRIGÉES : HYDROGEL_PACK=75, VELVETFRUIT_EXTRACT=400
     (la vraie limite Prosperity 4, pas 200 comme dans la version R3)
  2. Paramètres ré-optimisés sur les 3 jours historiques R4 via grid search
     avec validation IS/OOS (chemin train+test sur les 3 jours, exact=3)

Backtest historique sur les 3 jours R4 : +196 660 XIRECS
  vs baseline non-tunée (params Alexandre R3 + LIMIT corrigés) = +188 126
  → gain net de +8 534 XIRECS (+4.5 %) grâce à la calibration R4-spécifique

Conventions Prosperity 4
------------------------
- Order.quantity > 0 = BUY, < 0 = SELL
- OrderDepth.sell_orders : valeurs négatives
- traderData : seule persistance entre ticks
- Devise : XIRECS

Paramètres choisis (issus du grid search tunable, top-1)
--------------------------------------------------------
HYDROGEL_PACK :
  fair_value=9997 (mesuré : moy=9994, médiane=9999 ; 9997 = compromis)
  entry_threshold=22 (vs 24 R3 : entrée plus fréquente)
  pyramid_step=3 (inchangé R3)
  position limit=75

VELVETFRUIT_EXTRACT :
  fair_value=5245 (mesuré : moy=5247, médiane=5247 ; 5245 = sous-évalué légèrement)
  entry_threshold=12 (vs 15 R3 : entrée plus fréquente)
  pyramid_step=1 (vs 2 R3 : paliers plus fins)
  position limit=400

Forced close : à 95 % du round (timestamp >= 950000) → liquidation totale,
plus aucune nouvelle entrée. Évite MtM perdant en fin de session.
"""

import json

from datamodel import Order, TradingState

# ============================================================
# Configuration par produit — params optimaux issus du grid R4
# ============================================================
PRODUCT_CONFIG = {
    "HYDROGEL_PACK": {
        "fair_value": 9_997,  # tuné R4 (vs 9999 baseline)
        "entry_threshold": 22,  # tuné R4 (vs 24 baseline)
        "exit_threshold": 4,
        "pyramid_step": 3,
        "max_pyramid_levels": 5,
        "base_buy_units": 24,
        "buy_units_multiplier": 1.6,
        "sell_step": 1,
        "base_sell_units": 10,
        "sell_units_multiplier": 1.2,
        "max_units_held": 75,  # P4 official limit
        "limit": 75,
    },
    "VELVETFRUIT_EXTRACT": {
        "fair_value": 5_245,  # tuné R4 (vs 5250 baseline)
        "entry_threshold": 12,  # tuné R4 (vs 15 baseline)
        "exit_threshold": 2,
        "pyramid_step": 1,  # tuné R4 (vs 2 baseline)
        "max_pyramid_levels": 5,
        "base_buy_units": 24,
        "buy_units_multiplier": 1.6,
        "sell_step": 1,
        "base_sell_units": 10,
        "sell_units_multiplier": 1.2,
        "max_units_held": 400,  # P4 official limit
        "limit": 400,
    },
}

# Si timestamp > 0.95 du round → on ferme tout, on ne rouvre plus.
END_TS_FRACTION = 0.95
ROUND_TIMESTAMP_MAX = 1_000_000


# ============================================================
# Helpers de pyramidage et sortie progressive
# ============================================================
def units_for_level(base: int, multiplier: float, level: int) -> int:
    return max(1, int(base * (multiplier**level)))


def buy_target_position(distance_below_fair: float, cfg: dict) -> int:
    target = 0
    for level in range(cfg["max_pyramid_levels"]):
        thr = cfg["entry_threshold"] + level * cfg["pyramid_step"]
        if distance_below_fair >= thr:
            target += units_for_level(cfg["base_buy_units"], cfg["buy_units_multiplier"], level)
    return min(target, cfg["max_units_held"])


def short_target_position(distance_above_fair: float, cfg: dict) -> int:
    target = 0
    for level in range(cfg["max_pyramid_levels"]):
        thr = cfg["entry_threshold"] + level * cfg["pyramid_step"]
        if distance_above_fair >= thr:
            target += units_for_level(cfg["base_buy_units"], cfg["buy_units_multiplier"], level)
    return min(target, cfg["max_units_held"])


def close_target_long(distance_below_fair: float, peak_long: int, cfg: dict) -> int:
    if peak_long <= 0 or distance_below_fair <= cfg["exit_threshold"]:
        return 0
    farthest = cfg["exit_threshold"] + (cfg["max_pyramid_levels"] - 1) * cfg["sell_step"]
    if distance_below_fair > farthest:
        return peak_long
    units_to_sell = 0
    for level in range(cfg["max_pyramid_levels"]):
        thr = cfg["exit_threshold"] + (cfg["max_pyramid_levels"] - 1 - level) * cfg["sell_step"]
        if distance_below_fair <= thr:
            units_to_sell += units_for_level(
                cfg["base_sell_units"], cfg["sell_units_multiplier"], level
            )
    return max(0, peak_long - units_to_sell)


def close_target_short(distance_above_fair: float, peak_short: int, cfg: dict) -> int:
    return close_target_long(distance_above_fair, peak_short, cfg)


# ============================================================
# Persistance traderData
# ============================================================
def safe_load_trader_data(raw: str, products: list[str]) -> dict:
    if not raw:
        return {p: {"peak_long": 0, "peak_short": 0} for p in products}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {p: {"peak_long": 0, "peak_short": 0} for p in products}
    out = {}
    for p in products:
        d = data.get(p, {})
        out[p] = {
            "peak_long": int(d.get("peak_long", 0)),
            "peak_short": int(d.get("peak_short", 0)),
        }
    return out


def dump_trader_data(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"))


# ============================================================
# Logique principale par produit
# ============================================================
def trade_product(
    product: str, state: TradingState, persist: dict, cfg: dict, force_close: bool
) -> list[Order]:
    orders: list[Order] = []
    if product not in state.order_depths:
        return orders
    od = state.order_depths[product]
    if not od.buy_orders or not od.sell_orders:
        return orders

    best_bid = max(od.buy_orders.keys())
    best_ask = min(od.sell_orders.keys())
    bid_vol = od.buy_orders[best_bid]
    ask_vol = abs(od.sell_orders[best_ask])

    position = state.position.get(product, 0)
    fair = cfg["fair_value"]
    max_u = min(cfg["limit"], cfg["max_units_held"])

    peak_long = max(int(persist["peak_long"]), max(position, 0))
    peak_short = max(int(persist["peak_short"]), max(-position, 0))

    # Forced close en fin de session
    if force_close:
        if position > 0:
            qty = min(position, bid_vol)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        elif position < 0:
            qty = min(-position, ask_vol)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        persist["peak_long"] = 0
        persist["peak_short"] = 0
        return orders

    # ===== LONG side =====
    if position >= 0:
        long_dist = max(0.0, fair - best_ask)
        long_close_dist = max(0.0, fair - best_bid)
        buy_tgt = min(buy_target_position(long_dist, cfg), max_u)

        if buy_tgt > position:
            cap = max_u - position
            qty = min(buy_tgt - position, ask_vol, cap)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                peak_long = max(peak_long, position + qty)
        elif position > 0:
            if best_bid >= fair:
                sell_tgt = 0
            else:
                sell_tgt = close_target_long(long_close_dist, peak_long, cfg)
            sell_tgt = min(sell_tgt, position)
            if position > sell_tgt:
                qty = min(position - sell_tgt, bid_vol)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

        if position == 0 and not orders:
            peak_long = 0

    # ===== SHORT side =====
    if position <= 0 and not orders:
        short_dist = max(0.0, best_bid - fair)
        short_close_dist = max(0.0, best_ask - fair)
        short_tgt = min(short_target_position(short_dist, cfg), max_u)

        if short_tgt > -position:
            cap = max_u + position
            qty = min(short_tgt - (-position), bid_vol, cap)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                peak_short = max(peak_short, -position + qty)
        elif position < 0:
            if best_ask <= fair:
                buy_back_tgt = 0
            else:
                buy_back_tgt = close_target_short(short_close_dist, peak_short, cfg)
            buy_back_tgt = min(buy_back_tgt, -position)
            if -position > buy_back_tgt:
                qty = min(-position - buy_back_tgt, ask_vol)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))

        if position == 0 and not orders:
            peak_short = 0

    persist["peak_long"] = min(max(peak_long, 0), max_u)
    persist["peak_short"] = min(max(peak_short, 0), max_u)
    return orders


# ============================================================
# Trader class — point d'entrée Prosperity
# ============================================================
class Trader:
    """Stratégie finale R4 — params tunés sur les 3 jours historiques."""

    LIMIT: dict[str, int] = {p: cfg["limit"] for p, cfg in PRODUCT_CONFIG.items()}

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        products = list(PRODUCT_CONFIG.keys())
        persist = safe_load_trader_data(getattr(state, "traderData", ""), products)

        force_close = state.timestamp >= int(END_TS_FRACTION * ROUND_TIMESTAMP_MAX)

        for product, cfg in PRODUCT_CONFIG.items():
            orders = trade_product(product, state, persist[product], cfg, force_close)
            if orders:
                result[product] = orders

        return result, 0, dump_trader_data(persist)
