"""
Round 3 – Finaler Trader (Prosperity 4)
Strategie: Market-Making VELVETFRUIT_EXTRACT + Passive VEV-Exposure via Limit-Orders
Quelle: P3-Referenz Frankfurt Hedgehogs, adaptiert auf P4-Daten
"""

import json
import math

from datamodel import Order, OrderDepth, TradingState

# ─── Vol-Smile-Fit (vol_smile_fit-22.json, alle 3 Tage) ─────────────────────
_C2 = 0.14790173662078787
_C1 = -0.008009967571477103
_C0 = 0.2348233160318135

UNDERLYING = "VELVETFRUIT_EXTRACT"
LIMIT_UNDERLYING = 600

# VEV-Strikes: nur passive Limit-Orders (kein aktives Scalping)
ALL_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
LIMIT_VEV = 200

TICKS_PER_DAY = 1_000  # timestamps 0..99900 step 100 → 1000 Ticks/Tag
TICK_STEP = 100


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    return S * _ncdf(d1) - K * _ncdf(d1 - sigma * sqT)


def _smile_iv(m_t: float) -> float:
    return _C2 * m_t * m_t + _C1 * m_t + _C0


def _tte(timestamp: int, day_offset: int) -> float:
    frac = max(0.0, min(1.0, (timestamp // TICK_STEP) / TICKS_PER_DAY))
    # tte_base_days=8 laut vol_smile_fit-22.json
    days = max(0.0, 8.0 - float(day_offset) - frac)
    return days / 365.0


class Trader:
    # ── Tunable ─────────────────────────────────────────────────────────────
    # Market-Making Underlying
    MM_HALF_SPREAD: int = 2  # +/- 2 um fair value
    MM_SIZE: int = 20  # Lots pro Seite
    MM_SKEW_FACTOR: float = 0.1  # pos-Skew: 1 lot pos → 0.1 tick shift
    MM_MAX_POS: int = 400  # Soft-Limit, ab dem aggressiv flattened

    # Passive VEV-Limits
    VEV_EDGE_MIN: float = 0.5  # mind. 0.5 Preis-Units unter/über fair
    VEV_SIZE: int = 5  # passive Lot-Größe

    # VEV aktives Exit (flach auf Timeout)
    VEV_MAX_HOLD: int = 800

    WARMUP_TICKS: int = 30

    def run(self, state: TradingState):
        td = self._load(state.traderData)
        day_offset = int(td.get("day_offset", 0))
        last_ts = int(td.get("last_ts", -1))
        if last_ts >= 0 and state.timestamp < last_ts:
            day_offset += 1
        td["day_offset"] = day_offset
        td["last_ts"] = int(state.timestamp)
        tick = int(td.get("tick", 0)) + 1
        td["tick"] = tick

        result: dict[str, list[Order]] = {}

        # ── 1. VELVETFRUIT_EXTRACT Market-Making ────────────────────────────
        if UNDERLYING in state.order_depths:
            orders_mm = self._mm_underlying(
                state.order_depths[UNDERLYING], int(state.position.get(UNDERLYING, 0)), tick
            )
            if orders_mm:
                result[UNDERLYING] = orders_mm

        # ── 2. VEV Passive Limit-Orders (fair-value basiert) ────────────────
        if tick >= self.WARMUP_TICKS and UNDERLYING in state.order_depths:
            S = self._mid(state.order_depths[UNDERLYING])
            T = _tte(state.timestamp, day_offset)
            if S and S > 0 and T > 1e-9:
                sqrt_T = math.sqrt(T)
                vev_entries = td.setdefault("vev_entries", {})

                for K in ALL_STRIKES:
                    name = f"VEV_{K}"
                    if name not in state.order_depths:
                        continue
                    depth = state.order_depths[name]
                    pos = int(state.position.get(name, 0))
                    entry_t = vev_entries.get(name)

                    # Timeout-Flatten
                    if pos != 0 and entry_t is not None:
                        if (tick - int(entry_t)) >= self.VEV_MAX_HOLD:
                            bb = max(depth.buy_orders) if depth.buy_orders else None
                            ba = min(depth.sell_orders) if depth.sell_orders else None
                            if bb and ba:
                                result[name] = [Order(name, bb if pos > 0 else ba, -pos)]
                                vev_entries.pop(name, None)
                                continue

                    m_t = math.log(K / S) / sqrt_T
                    iv = max(_smile_iv(m_t), 0.01)
                    fair = _bs_call(S, float(K), T, iv)
                    if fair <= 0.1:
                        continue

                    bb = max(depth.buy_orders) if depth.buy_orders else None
                    ba = min(depth.sell_orders) if depth.sell_orders else None

                    orders_vev = []
                    lim = LIMIT_VEV
                    sz = self.VEV_SIZE

                    # Passiv Bid: kaufe wenn Markt-Bid >= fair + edge
                    if ba is not None and (fair - ba) >= self.VEV_EDGE_MIN:
                        avail = lim - pos
                        q = min(sz, avail, abs(int(depth.sell_orders.get(ba, 0))))
                        if q > 0:
                            orders_vev.append(Order(name, ba, q))

                    # Passiv Ask: verkaufe wenn Markt-Ask <= fair - edge
                    if bb is not None and (bb - fair) >= self.VEV_EDGE_MIN:
                        avail = lim + pos  # pos negativ erlaubt
                        q = min(sz, avail, abs(int(depth.buy_orders.get(bb, 0))))
                        if q > 0:
                            orders_vev.append(Order(name, bb, -q))

                    if orders_vev:
                        result[name] = orders_vev
                        if pos == 0:
                            vev_entries[name] = tick

        return result, 0, self._dump(td)

    # ── Market-Making Helper ─────────────────────────────────────────────────
    def _mm_underlying(self, depth: OrderDepth, pos: int, tick: int) -> list[Order]:
        fair = self._mid(depth)
        if fair is None:
            return []

        # Inventory-Skew
        skew = round(self.MM_SKEW_FACTOR * pos)
        bid_px = int(fair) - self.MM_HALF_SPREAD - skew
        ask_px = int(fair) + self.MM_HALF_SPREAD - skew

        # Hard-flatten wenn Position zu groß
        orders = []
        lim = LIMIT_UNDERLYING
        if pos > self.MM_MAX_POS:
            # Aggressiv verkaufen
            bb = max(depth.buy_orders) if depth.buy_orders else ask_px
            orders.append(Order(UNDERLYING, bb, -(pos - self.MM_MAX_POS // 2)))
            return orders
        if pos < -self.MM_MAX_POS:
            ba = min(depth.sell_orders) if depth.sell_orders else bid_px
            orders.append(Order(UNDERLYING, ba, (-pos - self.MM_MAX_POS // 2)))
            return orders

        # Normale MM-Quotes
        bid_vol = min(self.MM_SIZE, lim - pos)
        ask_vol = min(self.MM_SIZE, lim + pos)
        if bid_vol > 0:
            orders.append(Order(UNDERLYING, bid_px, bid_vol))
        if ask_vol > 0:
            orders.append(Order(UNDERLYING, ask_px, -ask_vol))
        return orders

    @staticmethod
    def _mid(depth: OrderDepth) -> float | None:
        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        return None if bb is None or ba is None else (bb + ba) / 2.0

    @staticmethod
    def _load(s: str) -> dict:
        if not s:
            return {}
        try:
            d = json.loads(s)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _dump(td: dict) -> str:
        try:
            return json.dumps(td, separators=(",", ":"))
        except Exception:
            return "{}"
