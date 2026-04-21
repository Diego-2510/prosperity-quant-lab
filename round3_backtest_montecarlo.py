#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
round3_backtest_montecarlo.py
==============================

Prosperity-4 Round-3 Backtesting + Monte-Carlo + Grid-Search Framework.

Quellen:
- Autoritativ für P4-Regeln:   Prosperity.txt   (im Space)
- Referenz-Muster aus P3 (nur Architektur/Heuristik, KEINE Produktfakten):
  README.md und FrankfurtHedgehogs_polished.txt

Kernbausteine:
    1) file discovery / parsing (offizielle Prosperity-CSVs + generischer JSON-Fallback)
    2) datamodel compatibility layer (lokale Minimalversionen von OrderDepth/Order/TradingState)
    3) order book normalization
    4) prosperity fill engine (EXACT + APPROX, Sign-Konvention, Limit-Aggregation)
    5) trader wrapper (dynamischer Import falls vorhanden; sonst generischer Default-Trader)
    6) feature engineering
    7) Monte-Carlo path generation (block bootstrap -> residual bootstrap -> jump-aware)
    8) parameter registry + coarse-to-fine grid search
    9) metrics (profit, drawdown, VaR, CVaR, turnover, robust stability)
    10) pareto frontier + robust top-2 selection
    11) visualization
    12) artifact export

Ausführung:
    python round3_backtest_montecarlo.py --data-dir ./data --out-dir ./bt_out \
        [--trader path/to/trader.py] [--n-paths 1000] [--seed 42]

Keine externen Abhängigkeiten außer numpy, pandas, matplotlib.
Python 3.12 kompatibel.

SIGN-KONVENTION (P4, autoritativ aus Prosperity.txt):
    - sell_orders im OrderDepth haben NEGATIVE Volumina.
    - positive Order.quantity = Buy, negative Order.quantity = Sell.
    - Wenn aggregierte Buys ODER aggregierte Sells einer Seite eines Produkts
      das absolute Positionslimit verletzen würden, wird diese Seite komplett
      verworfen.
    - traderData ist die einzige Persistenz zwischen Iterationen.
    - Unausgeführte Reste bleiben bis zum Iterationsende als eigene Quotes.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import random
import re
import sys
import time
import traceback
import warnings
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =============================================================================
# 0. GLOBAL CONFIG / ASSUMPTIONS
# =============================================================================

# ASSUMPTION: Round-3-Produkt-Liste, Limits und Felder sind aus den verfügbaren
# Space-Dateien NICHT ableitbar. Daher wird das Framework produkt-agnostisch
# gebaut; Produkte + Limits werden aus den geladenen Daten bzw. aus einer
# optionalen CLI/Config-Datei gezogen. Kein Produktname aus P3 wird als P4-Fakt
# behandelt.
DEFAULT_POSITION_LIMIT = 50  # ASSUMPTION: Fallback nur wenn Daten kein Limit liefern.

# ASSUMPTION: Bei offiziellen Prosperity-CSVs liegt typischerweise ein
# Zeitraster von 100-ms-Ticks ("timestamp" in Schritten von 100) vor.
# Das Framework greift aber nur auf die numerische Reihenfolge zu.
TICK_STEP = 100

# Fill-Modus-Flag: wird zur Laufzeit gesetzt.
FILL_MODE = "APPROX"  # wird bei verfügbarem L1/L2-Orderbuch auf "EXACT" gehoben.


# =============================================================================
# 1. DATAMODEL COMPATIBILITY LAYER
# =============================================================================
# Minimalversionen, die mit dem offiziellen Prosperity-datamodel kompatibel sind.
# Wenn "datamodel" verfügbar ist (z. B. via IMC-Sandbox), wird es bevorzugt.

try:
    from datamodel import (  # type: ignore
        Order as _OfficialOrder,
        OrderDepth as _OfficialOrderDepth,
        TradingState as _OfficialTradingState,
        Trade as _OfficialTrade,
        Observation as _OfficialObservation,
    )

    Order = _OfficialOrder
    OrderDepth = _OfficialOrderDepth
    TradingState = _OfficialTradingState
    Trade = _OfficialTrade
    Observation = _OfficialObservation
    _HAS_OFFICIAL_DATAMODEL = True
except Exception:  # pragma: no cover
    _HAS_OFFICIAL_DATAMODEL = False

    @dataclass
    class Order:  # type: ignore[no-redef]
        symbol: str
        price: int
        quantity: int  # positive = buy, negative = sell

    class OrderDepth:  # type: ignore[no-redef]
        def __init__(self) -> None:
            # buy_orders: price -> positive volume
            # sell_orders: price -> NEGATIVE volume (P4-Konvention)
            self.buy_orders: Dict[int, int] = {}
            self.sell_orders: Dict[int, int] = {}

    @dataclass
    class Trade:  # type: ignore[no-redef]
        symbol: str
        price: int
        quantity: int
        buyer: str = ""
        seller: str = ""
        timestamp: int = 0

    @dataclass
    class Observation:  # type: ignore[no-redef]
        plainValueObservations: Dict[str, Any] = field(default_factory=dict)
        conversionObservations: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class TradingState:  # type: ignore[no-redef]
        traderData: str
        timestamp: int
        listings: Dict[str, Any]
        order_depths: Dict[str, OrderDepth]
        own_trades: Dict[str, List[Trade]]
        market_trades: Dict[str, List[Trade]]
        position: Dict[str, int]
        observations: Observation


# =============================================================================
# 2. FILE DISCOVERY / PARSING
# =============================================================================

# Matche beliebige Runden + beliebige Tages-Suffixe (z.B. day_0, day_-1, day_-1-6).
# Erste ganze Zahl nach 'day' ist der DAY; optionale weitere Zahlen werden als
# Datei-Version/Upload-Suffix betrachtet.
PRICES_FILE_RE = re.compile(
    r"prices?_round[_-]?(-?\d+)[_-]?day[_-]?(-?\d+)(?:[_-]\d+)?\.csv$", re.IGNORECASE
)
TRADES_FILE_RE = re.compile(
    r"trades?_round[_-]?(-?\d+)[_-]?day[_-]?(-?\d+)(?:[_-]\d+)?(?:_nn)?\.csv$", re.IGNORECASE
)
GENERIC_CSV_RE = re.compile(r".*\.csv$", re.IGNORECASE)


def discover_files(
    data_dir: Path,
    round_filter: Optional[int] = None,
) -> Dict[str, List[Path]]:
    """Finde Prosperity-Preis-/Trade-Dateien + generische Fallback-CSVs.

    Falls round_filter gesetzt ist, werden nur Dateien dieser Round zurueckgegeben.
    data_dir darf auch Einzeldatei oder nicht-existent sein."""
    prices: List[Path] = []
    trades: List[Path] = []
    others: List[Path] = []
    if not data_dir.exists():
        return {"prices": [], "trades": [], "others": []}
    iterator = [data_dir] if data_dir.is_file() else sorted(data_dir.rglob("*"))
    for p in iterator:
        if not p.is_file():
            continue
        name = p.name
        mp = PRICES_FILE_RE.search(name)
        mt = TRADES_FILE_RE.search(name)
        if mp:
            if round_filter is None or int(mp.group(1)) == round_filter:
                prices.append(p)
        elif mt:
            if round_filter is None or int(mt.group(1)) == round_filter:
                trades.append(p)
        elif GENERIC_CSV_RE.search(name):
            others.append(p)
    return {"prices": prices, "trades": trades, "others": others}


def _read_prosperity_csv(path: Path) -> pd.DataFrame:
    """Prosperity-CSVs nutzen ';' als Delimiter; fallback auf ','."""
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:  # wrong delimiter
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep=",")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_prices(paths: List[Path]) -> pd.DataFrame:
    """Lädt + konkateniert Preis-Snapshots. Standardspalten (offizielles Format):
    day, timestamp, product, bid_price_1..3, bid_volume_1..3,
    ask_price_1..3, ask_volume_1..3, mid_price, profit_and_loss.
    """
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        df = _read_prosperity_csv(p)
        if "day" not in df.columns:
            m = PRICES_FILE_RE.search(p.name)
            # Gruppe 2 = day (Gruppe 1 = round).
            df["day"] = int(m.group(2)) if m else 0
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    # ask_volume_x ist positiv; wir konvertieren spaeter in negative sell_orders.
    return out


def load_trades(paths: List[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        df = _read_prosperity_csv(p)
        if "day" not in df.columns:
            m = TRADES_FILE_RE.search(p.name)
            df["day"] = int(m.group(2)) if m else 0
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def prices_to_order_depths(prices_df: pd.DataFrame) -> Dict[Tuple[int, int], Dict[str, OrderDepth]]:
    """Konvertiere Preisspalten zu OrderDepth-Objekten pro (day, timestamp)."""
    out: Dict[Tuple[int, int], Dict[str, OrderDepth]] = {}
    if prices_df.empty:
        return out
    cols = prices_df.columns
    bid_price_cols = [c for c in cols if re.match(r"bid_price_\d+$", c)]
    bid_vol_cols = [c for c in cols if re.match(r"bid_volume_\d+$", c)]
    ask_price_cols = [c for c in cols if re.match(r"ask_price_\d+$", c)]
    ask_vol_cols = [c for c in cols if re.match(r"ask_volume_\d+$", c)]

    for row in prices_df.itertuples(index=False):
        d = dict(zip(cols, row))
        key = (int(d.get("day", 0)), int(d.get("timestamp", 0)))
        prod = str(d.get("product"))
        depth = OrderDepth()
        for pc, vc in zip(bid_price_cols, bid_vol_cols):
            px = d.get(pc)
            vol = d.get(vc)
            if pd.notna(px) and pd.notna(vol) and vol != 0:
                depth.buy_orders[int(px)] = int(vol)
        for pc, vc in zip(ask_price_cols, ask_vol_cols):
            px = d.get(pc)
            vol = d.get(vc)
            if pd.notna(px) and pd.notna(vol) and vol != 0:
                # Prosperity-Konvention: sell_orders sind negativ.
                depth.sell_orders[int(px)] = -abs(int(vol))
        out.setdefault(key, {})[prod] = depth
    return out


# =============================================================================
# 3. ORDER BOOK NORMALIZATION / MID / MICROPRICE
# =============================================================================


def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """(best_bid_px, best_bid_vol_pos, best_ask_px, best_ask_vol_pos)"""
    bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
    bbv = depth.buy_orders.get(bb) if bb is not None else None
    bav = abs(depth.sell_orders.get(ba)) if ba is not None else None
    return bb, bbv, ba, bav


def mid_price(depth: OrderDepth) -> Optional[float]:
    bb, _, ba, _ = best_bid_ask(depth)
    if bb is None or ba is None:
        return None
    return 0.5 * (bb + ba)


def micro_price(depth: OrderDepth) -> Optional[float]:
    bb, bbv, ba, bav = best_bid_ask(depth)
    if bb is None or ba is None or not bbv or not bav:
        return mid_price(depth)
    return (bb * bav + ba * bbv) / (bbv + bav)


# =============================================================================
# 4. PROSPERITY-KOMPATIBLE FILL ENGINE
# =============================================================================
#
# Regeln (Prosperity.txt):
# - Orders matchen sofort gegen resting Quotes mit passendem Preis.
# - Ausführung zu Preisen der resting Orders.
# - Unfilled Reste bleiben bis Ende der Iteration als eigene Quotes und
#   können vom Bot-Flow getroffen werden (APPROX: wir modellieren das
#   konservativ über nächsten Tick-Mid-Crossing).
# - Aggregiertes Limit-Check pro Seite: wenn Buys oder Sells eines Produkts
#   aggregiert das absolute Positionslimit reißen wuerden, werden ALLE Orders
#   dieser Seite verworfen.


@dataclass
class FillResult:
    product: str
    price: int
    quantity: int  # signed: +buy, -sell (vom Trader-Perspektive)
    timestamp: int


def _aggregate_side_limit_check(
    orders: List[Order], current_position: int, position_limit: int
) -> List[Order]:
    """Implementiert die Prosperity-Regel: wenn aggregierte Buys bzw. Sells
    das absolute Positionslimit verletzen wuerden, werden ALLE Orders dieser
    Seite dieses Produkts verworfen."""
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    buy_sum = sum(o.quantity for o in buys)
    sell_sum = sum(o.quantity for o in sells)  # negativ
    # Nach Aggregation darf current_position + buy_sum <= +limit sein.
    # Und current_position + sell_sum >= -limit sein.
    kept: List[Order] = []
    if buys:
        if current_position + buy_sum <= position_limit:
            kept.extend(buys)
        # else: ganze Buy-Seite verworfen
    if sells:
        if current_position + sell_sum >= -position_limit:
            kept.extend(sells)
        # else: ganze Sell-Seite verworfen
    return kept


def simulate_fills_one_tick(
    orders_by_product: Dict[str, List[Order]],
    depths: Dict[str, OrderDepth],
    position: Dict[str, int],
    position_limits: Dict[str, int],
    next_mid: Dict[str, Optional[float]],
    timestamp: int,
    fill_mode: str = "APPROX",
) -> Tuple[List[FillResult], Dict[str, int]]:
    """Simuliere ein komplettes Matching für einen Tick.

    Returns
    -------
    fills : list of FillResult (signed quantity aus Trader-Sicht)
    position_delta : per-product signed delta
    """
    all_fills: List[FillResult] = []
    delta: Dict[str, int] = defaultdict(int)

    for prod, orders in orders_by_product.items():
        pos_now = position.get(prod, 0)
        limit = position_limits.get(prod, DEFAULT_POSITION_LIMIT)

        # 1) Side-aggregated limit check (P4-Regel).
        orders = _aggregate_side_limit_check(orders, pos_now, limit)
        if not orders:
            continue

        depth = depths.get(prod)
        if depth is None:
            continue

        # Lokale Kopien der Gegenseite, damit wir Volumen abbauen können.
        sell_book = dict(depth.sell_orders)  # px -> negativ
        buy_book = dict(depth.buy_orders)  # px -> positiv

        working_pos = pos_now
        # Sortiere: Buy-Orders nach höchstem Preis zuerst (aggressiv),
        # Sell-Orders nach niedrigstem Preis zuerst (aggressiv).
        orders_sorted = sorted(
            orders, key=lambda o: (-o.price if o.quantity > 0 else o.price)
        )
        unfilled: List[Order] = []

        for o in orders_sorted:
            qty_left = o.quantity
            if qty_left > 0:
                # BUY: matche gegen sell_book mit px <= o.price, günstigste zuerst.
                for ask_px in sorted(sell_book.keys()):
                    if ask_px > o.price or qty_left <= 0:
                        break
                    avail = -sell_book[ask_px]  # positiv
                    # Hardlimit: working_pos + fill <= limit
                    max_by_limit = limit - working_pos
                    take = min(qty_left, avail, max(0, max_by_limit))
                    if take <= 0:
                        break
                    all_fills.append(
                        FillResult(prod, ask_px, +take, timestamp)
                    )
                    delta[prod] += +take
                    working_pos += take
                    sell_book[ask_px] = -(avail - take)
                    if sell_book[ask_px] == 0:
                        del sell_book[ask_px]
                    qty_left -= take
            elif qty_left < 0:
                # SELL: matche gegen buy_book mit px >= o.price, höchste zuerst.
                for bid_px in sorted(buy_book.keys(), reverse=True):
                    if bid_px < o.price or qty_left >= 0:
                        break
                    avail = buy_book[bid_px]  # positiv
                    max_by_limit = limit + working_pos  # working_pos kann positiv sein; Sell -> neue Pos = working_pos - take >= -limit
                    take = min(-qty_left, avail, max(0, max_by_limit))
                    if take <= 0:
                        break
                    all_fills.append(
                        FillResult(prod, bid_px, -take, timestamp)
                    )
                    delta[prod] += -take
                    working_pos -= take
                    buy_book[bid_px] = avail - take
                    if buy_book[bid_px] == 0:
                        del buy_book[bid_px]
                    qty_left += take

            # Reste bleiben als resting quote.
            if qty_left != 0:
                unfilled.append(Order(o.symbol, o.price, qty_left))

        # 2) Unfilled Reste: im APPROX-Modus vereinfachen wir via
        # "next-tick mid crossing": wenn der nächste Mid den Resting-Preis
        # aus Sicht des Traders vorteilhaft kreuzt, gehen wir von einem
        # Teil-Fill aus. EXACT wuerde ein komplettes Bot-Flow-Modell brauchen.
        nm = next_mid.get(prod)
        if nm is not None and unfilled and fill_mode == "APPROX":
            for o in unfilled:
                if o.quantity > 0 and nm < o.price - 0.5:
                    # Unser Bid wird vom Markt angenommen -> Fill zu o.price.
                    max_by_limit = limit - working_pos
                    take = min(o.quantity, max(0, max_by_limit))
                    if take > 0:
                        all_fills.append(FillResult(prod, o.price, +take, timestamp))
                        delta[prod] += +take
                        working_pos += take
                elif o.quantity < 0 and nm > o.price + 0.5:
                    max_by_limit = limit + working_pos
                    take = min(-o.quantity, max(0, max_by_limit))
                    if take > 0:
                        all_fills.append(FillResult(prod, o.price, -take, timestamp))
                        delta[prod] += -take
                        working_pos -= take

    return all_fills, dict(delta)


# =============================================================================
# 5. TRADER WRAPPER + DEFAULT TRADER
# =============================================================================


class DefaultMarketMaker:
    """
    Generischer, parametrisierter Market-Making-Trader.
    Kompatibel mit Prosperity: stateless, Persistenz nur via traderData.
    Dient als Default, wenn kein echter Trader-Code übergeben wird.
    """

    # Hook-Name, damit Grid Search weiß, welche Parameter existieren.
    # mathematical_low: untere Schranke/Floor fuer Fair Value (niemals -inf;
    # -1e6 wirkt de facto als "kein Floor" fuer Prosperity-Preisniveaus).
    PARAM_SPEC = {
        "fair_value_window": {"type": "int", "grid": [10, 20, 40, 80]},
        "spread_edge": {"type": "int", "grid": [1, 2, 3, 4]},
        "max_order_size": {"type": "int", "grid": [5, 10, 15, 20]},
        "inventory_skew": {"type": "float", "grid": [0.0, 0.25, 0.5, 1.0]},
        "mean_reversion_thr": {"type": "float", "grid": [1.0, 2.0, 4.0]},
        "mathematical_low": {"type": "float", "grid": [-1e6, 0.0, 1.0]},
        "take_aggressive": {"type": "bool", "grid": [False, True]},
    }

    def __init__(self, params: Dict[str, Any], position_limits: Dict[str, int]):
        self.p = dict(params)
        self.position_limits = dict(position_limits)

    def _load_state(self, traderData: str) -> Dict[str, Any]:
        if not traderData:
            return {}
        try:
            return json.loads(traderData)
        except Exception:
            return {}

    def _dump_state(self, state: Dict[str, Any]) -> str:
        s = json.dumps(state, separators=(",", ":"))
        return s[:49_000]  # Prosperity-Limit ~ 50k.

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td = self._load_state(state.traderData)
        mids_hist: Dict[str, List[float]] = td.get("mids", {})
        orders_out: Dict[str, List[Order]] = {}

        for prod, depth in state.order_depths.items():
            mp = mid_price(depth)
            if mp is None:
                continue
            hist = mids_hist.get(prod, [])
            hist.append(mp)
            hist = hist[-max(5, self.p["fair_value_window"] * 2):]
            mids_hist[prod] = hist

            win = hist[-self.p["fair_value_window"]:]
            fv = float(np.mean(win)) if win else mp

            # Clip via mathematical_low (Floor). Nur greifen, wenn floor > -inf.
            ml = float(self.p["mathematical_low"])
            if math.isfinite(ml):
                fv = max(fv, ml)
            if not math.isfinite(fv):
                fv = mp

            pos = state.position.get(prod, 0)
            limit = self.position_limits.get(prod, DEFAULT_POSITION_LIMIT)
            skew = self.p["inventory_skew"] * (pos / max(1, limit))
            raw_bid = fv - self.p["spread_edge"] - skew
            raw_ask = fv + self.p["spread_edge"] - skew
            if not (math.isfinite(raw_bid) and math.isfinite(raw_ask)):
                continue
            bid_px = int(math.floor(raw_bid))
            ask_px = int(math.ceil(raw_ask))

            size = int(self.p["max_order_size"])
            buy_cap = max(0, limit - pos)
            sell_cap = max(0, limit + pos)

            ords: List[Order] = []

            bb, _, ba, _ = best_bid_ask(depth)
            # Take-Signal: mean reversion.
            if self.p["take_aggressive"] and ba is not None and (fv - ba) > self.p["mean_reversion_thr"]:
                take = min(size, buy_cap)
                if take > 0:
                    ords.append(Order(prod, int(ba), +take))
            if self.p["take_aggressive"] and bb is not None and (bb - fv) > self.p["mean_reversion_thr"]:
                take = min(size, sell_cap)
                if take > 0:
                    ords.append(Order(prod, int(bb), -take))

            # Make-Signal.
            mm_size = min(size, buy_cap)
            if mm_size > 0:
                ords.append(Order(prod, bid_px, +mm_size))
            mm_size = min(size, sell_cap)
            if mm_size > 0:
                ords.append(Order(prod, ask_px, -mm_size))

            if ords:
                orders_out[prod] = ords

        td["mids"] = mids_hist
        return orders_out, 0, self._dump_state(td)


def load_external_trader(path: Path) -> Optional[Callable]:
    """Lädt eine externe Trader-Klasse über import; erwartet `class Trader:` mit
    `run(self, state)`. Falls Import fehlschlägt, Rückgabe None (Default-Trader)."""
    if not path or not path.exists():
        return None
    spec_name = path.stem
    sys.path.insert(0, str(path.parent))
    try:
        mod = __import__(spec_name)
        TraderCls = getattr(mod, "Trader", None)
        if TraderCls is None:
            print(f"[WARN] {path} enthält keine Klasse 'Trader'.")
            return None
        return TraderCls
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Konnte externen Trader nicht laden: {e}")
        return None


# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================


def build_feature_frame(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Grund-Features pro (day, timestamp, product)."""
    if prices_df.empty:
        return pd.DataFrame()
    df = prices_df.copy()
    if "mid_price" not in df.columns:
        # Rekonstruiere aus bid_price_1 / ask_price_1
        bp = df.get("bid_price_1")
        ap = df.get("ask_price_1")
        if bp is not None and ap is not None:
            df["mid_price"] = (bp + ap) / 2
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    df["log_ret"] = (
        df.groupby("product")["mid_price"].apply(lambda s: np.log(s).diff()).reset_index(level=0, drop=True)
    )
    df["rolling_mean_20"] = (
        df.groupby("product")["mid_price"].rolling(20).mean().reset_index(level=0, drop=True)
    )
    df["rolling_std_20"] = (
        df.groupby("product")["mid_price"].rolling(20).std().reset_index(level=0, drop=True)
    )
    df["zscore_20"] = (df["mid_price"] - df["rolling_mean_20"]) / df["rolling_std_20"]
    df["spread"] = df.get("ask_price_1", np.nan) - df.get("bid_price_1", np.nan)
    return df


def feature_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for prod, grp in features.groupby("product"):
        rets = grp["log_ret"].dropna()
        if rets.empty:
            continue
        acf1 = float(rets.autocorr(lag=1)) if len(rets) > 2 else np.nan
        acf5 = float(rets.autocorr(lag=5)) if len(rets) > 5 else np.nan
        jumps = float((np.abs(rets) > 4 * rets.std()).mean()) if rets.std() else 0.0
        rows.append(
            dict(
                product=prod,
                n=len(rets),
                mean_ret=float(rets.mean()),
                std_ret=float(rets.std()),
                skew=float(rets.skew()),
                kurt=float(rets.kurt()),
                acf_lag1=acf1,
                acf_lag5=acf5,
                jump_frac=jumps,
                mean_spread=float(grp["spread"].dropna().mean()) if grp["spread"].notna().any() else np.nan,
            )
        )
    return pd.DataFrame(rows)


# =============================================================================
# 7. MONTE-CARLO PATH GENERATION
# =============================================================================


def _block_bootstrap_returns(rets: np.ndarray, n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if len(rets) == 0:
        return np.zeros(n)
    out = np.empty(n)
    filled = 0
    while filled < n:
        start = rng.integers(0, max(1, len(rets) - block_len + 1))
        blk = rets[start : start + block_len]
        take = min(block_len, n - filled)
        out[filled : filled + take] = blk[:take]
        filled += take
    return out


def _residual_bootstrap_returns(rets: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(rets) == 0:
        return np.zeros(n)
    mu = rets.mean()
    resid = rets - mu
    idx = rng.integers(0, len(resid), size=n)
    return mu + resid[idx]


def _add_jumps(
    series: np.ndarray, rng: np.random.Generator, jump_frac: float, jump_scale: float
) -> np.ndarray:
    if jump_frac <= 0:
        return series
    mask = rng.random(len(series)) < jump_frac
    jumps = rng.normal(0, jump_scale, size=len(series)) * mask
    return series + jumps


def generate_monte_carlo_paths(
    prices_df: pd.DataFrame,
    n_paths: int = 1000,
    seed: int = 42,
    block_len: int = 25,
) -> Dict[str, np.ndarray]:
    """
    Erzeuge fuer jedes Produkt (n_paths, T)-Matrix von Mid-Price-Pfaden.
    Reihenfolge: block bootstrap -> residual bootstrap -> jump-aware perturbation.
    Fuer gekoppelte Produkte: multivariate Reihenfolge via gemeinsame Indizes.
    """
    if prices_df.empty:
        return {}
    rng = np.random.default_rng(seed)

    # Pivot: index=(day,timestamp), columns=product, values=mid
    pv = (
        prices_df.pivot_table(index=["day", "timestamp"], columns="product", values="mid_price")
        .sort_index()
    )
    pv = pv.ffill().bfill()
    # Droppe Produkte, die komplett leer sind.
    pv = pv.dropna(axis=1, how="all")
    if pv.empty:
        return {}
    # Sicherheits-Fallback: verbleibende NaN mit Spalten-Mittelwert fuellen.
    pv = pv.fillna(pv.mean(numeric_only=True))
    products = list(pv.columns)
    T = len(pv)
    if T < 3:
        return {p: np.tile(pv[p].values, (n_paths, 1)) for p in products}

    # Gemeinsame Bootstrap-Indizes, um Cross-Asset-Korrelationen zu erhalten.
    rets = np.log(pv.values[1:] / pv.values[:-1])  # shape (T-1, P)
    out: Dict[str, np.ndarray] = {p: np.empty((n_paths, T)) for p in products}

    for i in range(n_paths):
        # Indizes fuer block bootstrap ueber die ZEIT-Achse -> Cross-Asset-Korr bleibt erhalten.
        sampled_idx = np.empty(T - 1, dtype=np.int64)
        filled = 0
        while filled < T - 1:
            start = rng.integers(0, max(1, len(rets) - block_len + 1))
            take = min(block_len, T - 1 - filled)
            sampled_idx[filled : filled + take] = np.arange(start, start + take)
            filled += take
        sampled = rets[sampled_idx]  # (T-1, P)

        # Mische 30% residual bootstrap fuer mehr Diversitaet.
        if rng.random() < 0.3:
            for k in range(sampled.shape[1]):
                sampled[:, k] = _residual_bootstrap_returns(rets[:, k], len(sampled), rng)

        # Jump perturbation.
        for k, prod in enumerate(products):
            std_k = rets[:, k].std() if rets.shape[0] > 1 else 0.0
            jump_frac = float((np.abs(rets[:, k]) > 4 * std_k).mean()) if std_k > 0 else 0.0
            sampled[:, k] = _add_jumps(sampled[:, k], rng, jump_frac, 3 * std_k)

        # Reintegriere zu Preisen, starte an historischem Startpreis.
        log_p0 = np.log(pv.values[0])
        log_path = np.vstack([log_p0, log_p0 + np.cumsum(sampled, axis=0)])
        prices = np.exp(log_path)
        for k, prod in enumerate(products):
            out[prod][i] = prices[:, k]

    return out


# =============================================================================
# 8. PARAMETER REGISTRY + GRID SEARCH
# =============================================================================


def extract_param_spec(trader_cls) -> Dict[str, Any]:
    """Versucht, aus einer externen Trader-Klasse ein PARAM_SPEC zu extrahieren.
    Erkennt: class attr PARAM_SPEC (wie hier), sonst CONFIG-dict, sonst nichts."""
    spec = getattr(trader_cls, "PARAM_SPEC", None)
    if isinstance(spec, dict):
        return spec
    cfg = getattr(trader_cls, "CONFIG", None)
    if isinstance(cfg, dict):
        # ASSUMPTION: CONFIG-Werte sind numerisch; wir bauen small-range-Grids
        # um jeden Default (+/- 1 Schritt).
        out = {}
        for k, v in cfg.items():
            if isinstance(v, bool):
                out[k] = {"type": "bool", "grid": [False, True]}
            elif isinstance(v, int):
                out[k] = {"type": "int", "grid": sorted(set([max(0, v - 1), v, v + 1]))}
            elif isinstance(v, float):
                out[k] = {"type": "float", "grid": sorted(set([v * 0.5, v, v * 1.5]))}
        return out
    return dict(DefaultMarketMaker.PARAM_SPEC)


def build_grid(spec: Dict[str, Any], focus_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    keys = focus_keys if focus_keys else list(spec.keys())
    grids = [spec[k]["grid"] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]


def coarse_to_fine_grid(
    spec: Dict[str, Any], run_batch: Callable[[List[Dict[str, Any]]], pd.DataFrame], top_k: int = 6
) -> pd.DataFrame:
    """Legacy auto-coarse-to-fine (triggert nur bei >500 Kombis). Bleibt fuer
    Kompatibilitaet erhalten; neuer expliziter Modus: two_stage_ctf()."""
    full = build_grid(spec)
    if len(full) <= 500:
        return run_batch(full)

    coarse_spec = {}
    for k, v in spec.items():
        grid = list(v["grid"])
        coarse_grid = grid[::2] if len(grid) > 2 else grid
        coarse_spec[k] = {"type": v["type"], "grid": coarse_grid}
    coarse = build_grid(coarse_spec)
    coarse_res = run_batch(coarse)

    top = coarse_res.nlargest(top_k, "objective_score")
    fine_combos = []
    for _, row in top.iterrows():
        local_spec = {}
        for k, v in spec.items():
            grid = list(v["grid"])
            center = row.get(k)
            if center in grid:
                i = grid.index(center)
                local_spec[k] = {
                    "type": v["type"],
                    "grid": grid[max(0, i - 1): i + 2],
                }
            else:
                local_spec[k] = v
        fine_combos.extend(build_grid(local_spec))
    seen = set()
    uniq = []
    for c in fine_combos:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    fine_res = run_batch(uniq)
    return pd.concat([coarse_res, fine_res], ignore_index=True).drop_duplicates()


def _refine_numeric_grid(values: Sequence[Any], n_interp: int = 2) -> List[Any]:
    """Verfeinere eine numerische Wertliste durch Einfuegen von n_interp
    aequidistanten Punkten zwischen benachbarten Werten. Fuer Bool/Kategorial
    unveraendert zurueckgeben."""
    vals = [v for v in values if v is not None]
    if not vals:
        return list(values)
    if all(isinstance(v, bool) for v in vals):
        return sorted(set(vals))
    try:
        nums = sorted(set(float(v) for v in vals))
    except (TypeError, ValueError):
        return sorted(set(vals), key=str)
    if len(nums) < 2:
        return nums
    out: List[float] = []
    for a, b in zip(nums[:-1], nums[1:]):
        out.append(a)
        step = (b - a) / (n_interp + 1)
        for i in range(1, n_interp + 1):
            out.append(a + step * i)
    out.append(nums[-1])
    # Runde Ints wieder zu Ints, wenn alle Eingabewerte int waren.
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        out = sorted(set(int(round(x)) for x in out))
    else:
        out = sorted(set(round(x, 6) for x in out))
    return out


def two_stage_ctf(
    spec: Dict[str, Any],
    run_batch_coarse: Callable[[List[Dict[str, Any]]], pd.DataFrame],
    run_batch_fine: Callable[[List[Dict[str, Any]]], pd.DataFrame],
    top_frac: float = 0.10,
    min_top: int = 4,
    max_top: int = 40,
    n_interp: int = 2,
) -> pd.DataFrame:
    """Expliziter Two-Stage-Coarse-to-Fine-Modus.

    Stufe 1: volles Grid ueber `spec`, ausgewertet mit run_batch_coarse
             (niedrige MC-Pfad-Anzahl -> schnelle Noise-Schaetzung).
    Stufe 2: selektiere Top-`top_frac` der Stufe 1, baue PRO KEY die
             Werte-Menge aus dieser Top-Region, verfeinere numerische Keys
             durch `_refine_numeric_grid`, bilde das Cross-Product und
             werte mit run_batch_fine (hoehere MC-Pfad-Anzahl) aus.

    Rueckgabe: konkateniertes DataFrame mit Spalte `stage` in {"coarse","fine"}.
    """
    coarse_combos = build_grid(spec)
    print(f"[CTF] Stage-1 coarse grid: {len(coarse_combos)} combos")
    coarse_df = run_batch_coarse(coarse_combos)
    if coarse_df.empty:
        return coarse_df.assign(stage="coarse")
    coarse_df = coarse_df.copy()
    coarse_df["stage"] = "coarse"

    n_top = int(max(min_top, min(max_top, math.ceil(len(coarse_df) * top_frac))))
    top = coarse_df.nlargest(n_top, "objective_score")
    print(f"[CTF] Stage-1 kept top {len(top)} rows (~{top_frac*100:.0f}%) for refinement")

    # Baue fine_spec: pro Key nimm die in Top-Region vorkommenden Werte
    # (plus optionale numerische Verfeinerung).
    fine_spec: Dict[str, Any] = {}
    for k, v in spec.items():
        if k not in top.columns:
            fine_spec[k] = v
            continue
        seen_vals = [x for x in top[k].tolist() if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if not seen_vals:
            fine_spec[k] = v
            continue
        if v.get("type") in ("int", "float"):
            fine_grid = _refine_numeric_grid(seen_vals, n_interp=n_interp)
        else:
            fine_grid = sorted(set(seen_vals), key=str)
        fine_spec[k] = {"type": v["type"], "grid": fine_grid}

    fine_combos = build_grid(fine_spec)
    # Dedupe gegen bereits evaluierte coarse-Kombis (exakter Parameter-Match).
    coarse_keys = set(
        tuple(sorted((k, coarse_df.iloc[i][k]) for k in spec.keys() if k in coarse_df.columns))
        for i in range(len(coarse_df))
    )
    fine_unique = []
    for c in fine_combos:
        key = tuple(sorted(c.items()))
        if key not in coarse_keys:
            fine_unique.append(c)
    print(f"[CTF] Stage-2 fine grid: {len(fine_unique)} new combos (of {len(fine_combos)} generated)")

    if not fine_unique:
        return coarse_df

    fine_df = run_batch_fine(fine_unique)
    if fine_df is None or fine_df.empty:
        return coarse_df
    fine_df = fine_df.copy()
    fine_df["stage"] = "fine"
    return pd.concat([coarse_df, fine_df], ignore_index=True)


# =============================================================================
# 9. BACKTEST CORE + METRICS
# =============================================================================


def infer_position_limits(prices_df: pd.DataFrame) -> Dict[str, int]:
    """ASSUMPTION: Ohne explizite P4-Round-3-Limit-Info leiten wir konservativ
    aus maximalem beobachteten Volumen ab (mind. DEFAULT_POSITION_LIMIT)."""
    if prices_df.empty:
        return {}
    limits: Dict[str, int] = {}
    for prod, grp in prices_df.groupby("product"):
        vol_cols = [c for c in prices_df.columns if "volume" in c]
        maxvol = float(grp[vol_cols].abs().max().max()) if vol_cols else 0
        limits[str(prod)] = max(DEFAULT_POSITION_LIMIT, int(math.ceil(maxvol)))
    return limits


def run_backtest_on_series(
    trader_cls,
    params: Dict[str, Any],
    mids_by_product: Dict[str, np.ndarray],
    historical_depths: Optional[Dict[Tuple[int, int], Dict[str, OrderDepth]]],
    ordered_keys: Optional[List[Tuple[int, int]]],
    position_limits: Dict[str, int],
    max_ticks: Optional[int] = None,
) -> Dict[str, Any]:
    """Führt einen Backtest aus.
    Wenn historical_depths + ordered_keys gegeben: EXACT-Modus (nutzt echte Orderbücher).
    Sonst: APPROX-Modus auf Mid-Serien (synthetisches Orderbuch Bid=mid-1, Ask=mid+1).
    """
    global FILL_MODE

    # Instantiate trader.
    if trader_cls is DefaultMarketMaker:
        trader = DefaultMarketMaker(params, position_limits)
    else:
        try:
            trader = trader_cls()
            # Setze Parameter dynamisch.
            for k, v in params.items():
                try:
                    setattr(trader, k, v)
                except Exception:
                    pass
            # Manche externen Trader nutzen CONFIG-dict:
            cfg = getattr(trader, "CONFIG", None)
            if isinstance(cfg, dict):
                for k, v in params.items():
                    if k in cfg:
                        cfg[k] = v
        except Exception:
            trader = DefaultMarketMaker(params, position_limits)

    position: Dict[str, int] = defaultdict(int)
    cash = 0.0
    realized_pnl_path: List[float] = []
    turnover = 0.0
    inventory_series: List[float] = []
    traderData = ""

    if historical_depths is not None and ordered_keys is not None and len(ordered_keys) > 1:
        FILL_MODE = "EXACT"
        keys = ordered_keys
        if max_ticks is not None and max_ticks > 0 and len(keys) > max_ticks:
            keys = keys[:max_ticks]
        # Precompute next-mid cache.
        all_prods = set()
        for _, d in historical_depths.items():
            all_prods.update(d.keys())
        next_mid_cache: Dict[int, Dict[str, Optional[float]]] = {}
        for i, k in enumerate(keys):
            nk = keys[i + 1] if i + 1 < len(keys) else None
            mids = {}
            if nk is not None:
                for p, dp in historical_depths.get(nk, {}).items():
                    mids[p] = mid_price(dp)
            next_mid_cache[i] = mids

        for i, k in enumerate(keys):
            depths = historical_depths[k]
            ts = k[1]
            state = TradingState(
                traderData=traderData,
                timestamp=ts,
                listings={},
                order_depths=depths,
                own_trades={},
                market_trades={},
                position=dict(position),
                observations=Observation(),
            )
            try:
                res = trader.run(state)
                if isinstance(res, tuple) and len(res) == 3:
                    orders_out, _conversions, traderData = res
                else:
                    orders_out = res if isinstance(res, dict) else {}
            except Exception:
                orders_out = {}
            fills, delta = simulate_fills_one_tick(
                orders_out, depths, dict(position), position_limits, next_mid_cache[i], ts, "EXACT"
            )
            for f in fills:
                cash -= f.price * f.quantity  # buy -> cash runter
                turnover += abs(f.price * f.quantity)
                position[f.product] += f.quantity
            # Mark-to-market via aktuellem Mid.
            mtm = 0.0
            for p, pos in position.items():
                dp = depths.get(p)
                if dp is not None:
                    m = mid_price(dp)
                    if m is not None:
                        mtm += pos * m
            realized_pnl_path.append(cash + mtm)
            inventory_series.append(sum(abs(v) for v in position.values()))
    else:
        FILL_MODE = "APPROX"
        # Synthetisches Orderbuch aus Mid.
        products = list(mids_by_product.keys())
        # Filtere NaNs pro Produkt
        clean: Dict[str, np.ndarray] = {}
        for p in products:
            arr = np.asarray(mids_by_product[p], dtype=float)
            if arr.size == 0 or np.all(np.isnan(arr)):
                continue
            # forward-fill NaNs
            mask = np.isnan(arr)
            if mask.any():
                idx = np.where(~mask, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]
                clean[p] = arr
        products = list(clean.keys())
        T = min(len(clean[p]) for p in products) if products else 0
        if max_ticks is not None and max_ticks > 0:
            T = min(T, max_ticks)
        for t in range(T):
            depths: Dict[str, OrderDepth] = {}
            for p in products:
                m = float(clean[p][t])
                if not math.isfinite(m):
                    continue
                od = OrderDepth()
                od.buy_orders[int(math.floor(m - 1))] = 30
                od.sell_orders[int(math.ceil(m + 1))] = -30
                depths[p] = od
            nm = {
                p: (float(clean[p][t + 1]) if t + 1 < T else None) for p in products
            }
            state = TradingState(
                traderData=traderData,
                timestamp=t * TICK_STEP,
                listings={},
                order_depths=depths,
                own_trades={},
                market_trades={},
                position=dict(position),
                observations=Observation(),
            )
            try:
                res = trader.run(state)
                if isinstance(res, tuple) and len(res) == 3:
                    orders_out, _conv, traderData = res
                else:
                    orders_out = res if isinstance(res, dict) else {}
            except Exception:
                orders_out = {}
            fills, delta = simulate_fills_one_tick(
                orders_out, depths, dict(position), position_limits, nm, t * TICK_STEP, "APPROX"
            )
            for f in fills:
                cash -= f.price * f.quantity
                turnover += abs(f.price * f.quantity)
                position[f.product] += f.quantity
            mtm = sum(position.get(p, 0) * float(clean[p][t]) for p in products)
            realized_pnl_path.append(cash + mtm)
            inventory_series.append(sum(abs(v) for v in position.values()))

    pnl = np.array(realized_pnl_path) if realized_pnl_path else np.array([0.0])
    final_pnl = float(pnl[-1])
    returns = np.diff(pnl, prepend=0.0)
    max_dd = float((np.maximum.accumulate(pnl) - pnl).max()) if len(pnl) else 0.0
    return dict(
        final_pnl=final_pnl,
        mean_step_pnl=float(returns.mean()) if len(returns) else 0.0,
        std_step_pnl=float(returns.std()) if len(returns) > 1 else 0.0,
        max_drawdown=max_dd,
        hit_rate=float((returns > 0).mean()) if len(returns) else 0.0,
        turnover=float(turnover),
        inventory_std=float(np.std(inventory_series)) if inventory_series else 0.0,
        equity_curve=pnl.tolist(),
    )


# =============================================================================
# 10. METRICS OVER MONTE-CARLO PATHS + ROBUST SELECTION
# =============================================================================


def aggregate_path_metrics(
    per_path_profits: np.ndarray, risk_lambda: float
) -> Dict[str, float]:
    if len(per_path_profits) == 0:
        return {}
    mean_p = float(np.mean(per_path_profits))
    med_p = float(np.median(per_path_profits))
    std_p = float(np.std(per_path_profits))
    var_p = float(np.var(per_path_profits))
    var_5 = float(np.quantile(per_path_profits, 0.05))
    cvar_5 = float(per_path_profits[per_path_profits <= var_5].mean()) if np.any(per_path_profits <= var_5) else var_5
    worst = float(per_path_profits.min())

    # Normalisierung (gegen Cross-Parameter-Vergleichbarkeit innerhalb desselben Grids).
    norm_mean = mean_p
    norm_var = var_p
    score = norm_mean - risk_lambda * norm_var
    return dict(
        total_profit=float(np.sum(per_path_profits)),
        mean_profit=mean_p,
        median_profit=med_p,
        profit_std=std_p,
        variance=var_p,
        VaR_5=var_5,
        CVaR_5=cvar_5,
        worst_path_profit=worst,
        objective_score=score,
    )


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, maximize_y: bool = True) -> pd.DataFrame:
    """Pareto: wir wollen mean_profit MAX und variance MIN.
    Hier x=variance (min), y=mean_profit (max)."""
    if df.empty:
        return df
    s = df.sort_values([x_col, y_col], ascending=[True, not maximize_y])
    front = []
    best_y = -np.inf if maximize_y else np.inf
    for _, r in s.iterrows():
        y = r[y_col]
        if (maximize_y and y > best_y) or (not maximize_y and y < best_y):
            front.append(r)
            best_y = y
    return pd.DataFrame(front)


def pick_top2_robust(full_results: pd.DataFrame) -> pd.DataFrame:
    """Robuste Top-2-Auswahl: Pareto + lokale Stabilität.
    Stabilität = mittleres objective_score der Nachbarn im Grid.
    """
    if full_results.empty:
        return full_results
    df = full_results.copy()
    # Normalisierung fuer robuste Metrik.
    df["obj_norm"] = (df["objective_score"] - df["objective_score"].mean()) / (
        df["objective_score"].std() + 1e-9
    )
    # Nachbar-Score: Mittelwert der nearest-neighbors in Parameterraum (euklidisch
    # ueber numerische Parameter).
    num_cols = [
        c for c in df.columns
        if c not in {
            "objective_score", "obj_norm", "total_profit", "mean_profit", "median_profit",
            "profit_std", "variance", "VaR_5", "CVaR_5", "worst_path_profit", "max_drawdown",
            "hit_rate", "turnover", "inventory_std",
        } and pd.api.types.is_numeric_dtype(df[c])
    ]
    if num_cols:
        X = df[num_cols].values.astype(float)
        # z-score
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Xn = (X - mu) / sd
        stab = np.empty(len(df))
        K = min(5, len(df) - 1)
        for i in range(len(df)):
            d = np.linalg.norm(Xn - Xn[i], axis=1)
            idx = np.argsort(d)[1 : K + 1]
            stab[i] = df["obj_norm"].values[idx].mean() if len(idx) else df["obj_norm"].values[i]
        df["neighborhood_stability"] = stab
    else:
        df["neighborhood_stability"] = df["obj_norm"]

    df["robust_score"] = 0.5 * df["obj_norm"] + 0.5 * df["neighborhood_stability"]
    return df.nlargest(2, "robust_score")


# =============================================================================
# 11. VISUALIZATION
# =============================================================================


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_equity_curve(equity: List[float], out: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity, lw=1.2)
    ax.set_title("Historical Equity Curve (Best Params)")
    ax.set_xlabel("tick"); ax.set_ylabel("PnL")
    save_plot(fig, out)


def plot_fan_chart(per_path_equity: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    qs = np.quantile(per_path_equity, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    x = np.arange(per_path_equity.shape[1])
    ax.fill_between(x, qs[0], qs[4], alpha=0.15, label="5-95%")
    ax.fill_between(x, qs[1], qs[3], alpha=0.3, label="25-75%")
    ax.plot(x, qs[2], lw=1.4, label="median")
    ax.set_title("Monte-Carlo Fan Chart (Best Params)")
    ax.legend(); ax.set_xlabel("tick"); ax.set_ylabel("PnL")
    save_plot(fig, out)


def plot_final_pnl_dist(finals: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(finals, bins=40)
    ax.set_title("Distribution of Final PnL across MC paths")
    save_plot(fig, out)


def plot_profit_vs_variance(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["variance"], df["mean_profit"], s=12, alpha=0.6)
    ax.set_xlabel("variance"); ax.set_ylabel("mean_profit")
    ax.set_title("Profit vs Variance (all param combos)")
    save_plot(fig, out)


def plot_pareto(df_all: pd.DataFrame, df_pareto: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_all["variance"], df_all["mean_profit"], s=10, alpha=0.3, label="all")
    ax.scatter(df_pareto["variance"], df_pareto["mean_profit"], s=30, color="red", label="pareto")
    ax.set_xlabel("variance"); ax.set_ylabel("mean_profit"); ax.legend()
    ax.set_title("Pareto Frontier")
    save_plot(fig, out)


def plot_heatmap_top2(df: pd.DataFrame, keys: Tuple[str, str], out: Path):
    if keys[0] not in df.columns or keys[1] not in df.columns:
        return
    piv = df.pivot_table(index=keys[0], columns=keys[1], values="objective_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([str(c) for c in piv.columns], rotation=45)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([str(i) for i in piv.index])
    ax.set_xlabel(keys[1]); ax.set_ylabel(keys[0])
    ax.set_title(f"Objective Score Heatmap: {keys[0]} vs {keys[1]}")
    fig.colorbar(im, ax=ax)
    save_plot(fig, out)


def plot_sensitivity_math_low(df: pd.DataFrame, out: Path):
    if "mathematical_low" not in df.columns:
        return
    grp = df.groupby("mathematical_low")["objective_score"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(len(grp)), grp["mean"], yerr=grp["std"], marker="o")
    ax.set_xticks(range(len(grp))); ax.set_xticklabels([f"{v:g}" for v in grp["mathematical_low"]])
    ax.set_title("Sensitivity: mathematical_low")
    ax.set_xlabel("mathematical_low"); ax.set_ylabel("objective_score")
    save_plot(fig, out)


def plot_top_ranking(df: pd.DataFrame, out: Path, n: int = 20):
    top = df.nlargest(n, "objective_score").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top)), top["objective_score"])
    ax.set_title(f"Top-{n} Parameter Combinations by objective_score")
    ax.set_xlabel("rank"); ax.set_ylabel("objective_score")
    save_plot(fig, out)


def plot_stability(top_k_path_profits: List[np.ndarray], labels: List[str], out: Path):
    if not top_k_path_profits:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(top_k_path_profits, labels=labels, showfliers=False)
    ax.set_title("Profit Distribution across MC paths — Top-K Params")
    plt.xticks(rotation=45, ha="right")
    save_plot(fig, out)


# =============================================================================
# 12. MAIN PIPELINE
# =============================================================================


def _single_day_keys(
    ordered_keys: List[Tuple[int, int]], exact_days: int
) -> List[Tuple[int, int]]:
    """Begrenze EXACT-Modus auf die ersten exact_days Tage (deterministisch)."""
    if not ordered_keys or exact_days <= 0:
        return []
    days_seen: List[int] = []
    picked: List[Tuple[int, int]] = []
    for k in ordered_keys:
        if k[0] not in days_seen:
            if len(days_seen) >= exact_days:
                break
            days_seen.append(k[0])
        picked.append(k)
    return picked


def run_pipeline(
    data_dir: Path,
    out_dir: Path,
    trader_path: Optional[Path],
    n_paths: int,
    seed: int,
    risk_lambda: float,
    round_filter: Optional[int] = None,
    max_ticks: Optional[int] = None,
    exact_days: int = 1,
    eval_paths: int = 50,
    eval_paths_final: int = 200,
    skip_hist_in_grid: bool = True,
    ctf_enabled: bool = False,
    ctf_eval_coarse: int = 10,
    ctf_eval_fine: int = 100,
    ctf_top_frac: float = 0.10,
    ctf_n_interp: int = 2,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # -- Discover & load.
    files = discover_files(data_dir, round_filter=round_filter)
    print(f"[INFO] prices={len(files['prices'])} trades={len(files['trades'])} others={len(files['others'])}")

    prices_df = load_prices(files["prices"])
    trades_df = load_trades(files["trades"])

    # Fallback: wenn kein Round-3-Match, akzeptiere generische CSVs mit
    # Spalten {timestamp, product, mid_price}.
    if prices_df.empty and files["others"]:
        frames = []
        for p in files["others"]:
            try:
                df = _read_prosperity_csv(p)
                if {"timestamp", "product"}.issubset(df.columns):
                    if "mid_price" not in df.columns and {"bid_price_1", "ask_price_1"}.issubset(df.columns):
                        df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
                    df["day"] = df.get("day", 0)
                    frames.append(df)
            except Exception:
                continue
        if frames:
            prices_df = pd.concat(frames, ignore_index=True)

    if prices_df.empty:
        # ASSUMPTION: Daten fehlen. Erzeuge synthetischen 3-Pfad-Datensatz
        # damit der Lauf demonstrativ ablaeuft und der Monte-Carlo-Teil testbar ist.
        print("[WARN] Keine Prosperity-Daten gefunden. Erzeuge synthetischen Fallback (3 Tage, 1 Produkt).")
        rng = np.random.default_rng(seed)
        days = []
        T = 1000
        for d in range(3):
            mids = 100 + np.cumsum(rng.normal(0, 0.3, size=T))
            days.append(pd.DataFrame({
                "day": d,
                "timestamp": np.arange(T) * TICK_STEP,
                "product": "SYNTH_PRODUCT",
                "mid_price": mids,
                "bid_price_1": np.floor(mids - 1).astype(int),
                "ask_price_1": np.ceil(mids + 1).astype(int),
                "bid_volume_1": 30,
                "ask_volume_1": 30,
            }))
        prices_df = pd.concat(days, ignore_index=True)

    # -- Features.
    features = build_feature_frame(prices_df)
    diag = feature_diagnostics(features)
    diag.to_csv(out_dir / "feature_diagnostics.csv", index=False)

    # -- Position limits.
    position_limits = infer_position_limits(prices_df)
    print(f"[INFO] inferred position limits: {position_limits}")

    # -- Historical depths (EXACT falls moeglich).
    historical_depths = prices_to_order_depths(prices_df)
    ordered_keys_all = sorted(historical_depths.keys())
    # Default: EXACT-Modus auf exact_days Tage begrenzen (deutlich schneller).
    ordered_keys = _single_day_keys(ordered_keys_all, exact_days)
    if max_ticks is not None and max_ticks > 0 and len(ordered_keys) > max_ticks:
        ordered_keys = ordered_keys[:max_ticks]
    have_exact = len(ordered_keys) > 1 and any(
        any(d.buy_orders or d.sell_orders for d in historical_depths.get(k, {}).values())
        for k in ordered_keys
    )
    print(
        f"[INFO] EXACT-fill: available={have_exact} keys={len(ordered_keys)} "
        f"(of {len(ordered_keys_all)}) exact_days={exact_days} max_ticks={max_ticks}"
    )

    # -- Trader.
    trader_cls = load_external_trader(trader_path) if trader_path else None
    if trader_cls is None:
        trader_cls = DefaultMarketMaker
        print("[INFO] Verwende DefaultMarketMaker (kein externer Trader).")
    else:
        print(f"[INFO] Verwende externen Trader: {trader_cls.__name__}")

    # -- Parameter registry.
    spec = extract_param_spec(trader_cls)
    print(f"[INFO] Parameter registry keys: {list(spec.keys())}")

    # -- Monte-Carlo paths (truncated to max_ticks if set).
    print(f"[INFO] Generating {n_paths} Monte-Carlo paths ...")
    mc_paths = generate_monte_carlo_paths(prices_df, n_paths=n_paths, seed=seed)
    if max_ticks is not None and max_ticks > 0:
        mc_paths = {p: arr[:, :max_ticks] for p, arr in mc_paths.items()}
    products = list(mc_paths.keys())
    mc_summary_rows = []
    for p, arr in mc_paths.items():
        mc_summary_rows.append(dict(
            product=p,
            n_paths=arr.shape[0],
            horizon=arr.shape[1],
            mean_final=float(arr[:, -1].mean()),
            std_final=float(arr[:, -1].std()),
        ))
    pd.DataFrame(mc_summary_rows).to_csv(out_dir / "monte_carlo_summary.csv", index=False)

    # -- Backtest-Runner (single param, over MC paths).
    T_path = min(arr.shape[1] for arr in mc_paths.values()) if mc_paths else 0
    hist_mids = {p: prices_df[prices_df["product"] == p]["mid_price"].values for p in products}

    def _make_batch_runner(n_eval: int) -> Callable[[List[Dict[str, Any]]], pd.DataFrame]:
        n_eval = max(1, min(n_paths, n_eval))

        def run_backtest_for_params(params: Dict[str, Any]) -> Dict[str, float]:
            if not skip_hist_in_grid:
                hist_res = run_backtest_on_series(
                    trader_cls, params,
                    mids_by_product=hist_mids,
                    historical_depths=historical_depths if have_exact else None,
                    ordered_keys=ordered_keys if have_exact else None,
                    position_limits=position_limits,
                    max_ticks=max_ticks,
                )
            else:
                hist_res = dict(final_pnl=0.0, max_drawdown=0.0, hit_rate=0.0,
                                turnover=0.0, inventory_std=0.0, equity_curve=[0.0])
            profits: List[float] = []
            rng = np.random.default_rng(seed + hash(tuple(sorted(params.items()))) % 2**31)
            idx = rng.choice(n_paths, size=n_eval, replace=False)
            for i in idx:
                mids = {p: mc_paths[p][i] for p in products}
                r = run_backtest_on_series(
                    trader_cls, params, mids_by_product=mids,
                    historical_depths=None, ordered_keys=None,
                    position_limits=position_limits,
                    max_ticks=max_ticks,
                )
                profits.append(r["final_pnl"])
            profits_arr = np.array(profits)
            m = aggregate_path_metrics(profits_arr, risk_lambda)
            m.update(dict(
                max_drawdown=hist_res["max_drawdown"],
                hit_rate=hist_res["hit_rate"],
                turnover=hist_res["turnover"],
                inventory_std=hist_res["inventory_std"],
                historical_final_pnl=hist_res["final_pnl"],
                n_eval_paths=n_eval,
            ))
            return m

        def run_batch(combos: List[Dict[str, Any]]) -> pd.DataFrame:
            rows = []
            for i, c in enumerate(combos):
                try:
                    metrics = run_backtest_for_params(c)
                except Exception as e:
                    print(f"[ERR] combo {i} failed: {e}")
                    continue
                row = dict(c); row.update(metrics)
                rows.append(row)
                if (i + 1) % 25 == 0:
                    print(f"  [grid] {i + 1}/{len(combos)} evaluated (n_eval={n_eval})")
            return pd.DataFrame(rows)

        return run_batch

    # -- Grid Search.
    if ctf_enabled:
        print(
            f"[INFO] Running two-stage Coarse-to-Fine grid search: "
            f"coarse_eval={ctf_eval_coarse}, fine_eval={ctf_eval_fine}, "
            f"top_frac={ctf_top_frac}, n_interp={ctf_n_interp}"
        )
        run_batch_coarse = _make_batch_runner(ctf_eval_coarse)
        run_batch_fine = _make_batch_runner(ctf_eval_fine)
        full_results = two_stage_ctf(
            spec, run_batch_coarse, run_batch_fine,
            top_frac=ctf_top_frac, n_interp=ctf_n_interp,
        )
    else:
        print(f"[INFO] Running single-stage grid search (eval_paths={eval_paths})")
        run_batch = _make_batch_runner(eval_paths)
        full_results = coarse_to_fine_grid(spec, run_batch, top_k=6)
    full_results.to_csv(out_dir / "full_grid_results.csv", index=False)
    print(f"[INFO] grid results: {len(full_results)} rows")

    # -- Pareto + robust Top-2.
    pareto = pareto_frontier(full_results.dropna(subset=["variance", "mean_profit"]),
                             x_col="variance", y_col="mean_profit", maximize_y=True)
    pareto.to_csv(out_dir / "pareto_front.csv", index=False)

    top2 = pick_top2_robust(full_results)
    top2.to_csv(out_dir / "top_parameter_pairs.csv", index=False)
    print("[INFO] Robust Top-2 parameter sets:")
    print(top2.to_string(index=False))

    # -- Visualisierung.
    # Best-Param historische Equity (einmaliger EXACT-Lauf, kein Grid-Overhead).
    if not top2.empty:
        best = top2.iloc[0].to_dict()
        best_params = {k: best[k] for k in spec.keys() if k in best}
        hist_res = run_backtest_on_series(
            trader_cls, best_params,
            mids_by_product=hist_mids,
            historical_depths=historical_depths if have_exact else None,
            ordered_keys=ordered_keys if have_exact else None,
            position_limits=position_limits,
            max_ticks=max_ticks,
        )
        plot_equity_curve(hist_res["equity_curve"], out_dir / "plot_equity_curve.png")

        # Fan chart + final pnl dist fuer best params.
        path_curves = []
        finals = []
        for i in range(min(eval_paths_final, n_paths)):
            mids = {p: mc_paths[p][i] for p in products}
            r = run_backtest_on_series(
                trader_cls, best_params, mids_by_product=mids,
                historical_depths=None, ordered_keys=None,
                position_limits=position_limits,
                max_ticks=max_ticks,
            )
            path_curves.append(r["equity_curve"])
            finals.append(r["final_pnl"])
        minlen = min(len(c) for c in path_curves)
        eq_matrix = np.array([c[:minlen] for c in path_curves])
        plot_fan_chart(eq_matrix, out_dir / "plot_mc_fan_chart.png")
        plot_final_pnl_dist(np.array(finals), out_dir / "plot_final_pnl_dist.png")

    plot_profit_vs_variance(full_results, out_dir / "plot_profit_vs_variance.png")
    plot_pareto(full_results, pareto, out_dir / "plot_pareto.png")
    plot_top_ranking(full_results, out_dir / "plot_top_ranking.png")
    plot_sensitivity_math_low(full_results, out_dir / "plot_sensitivity_mathematical_low.png")

    # Heatmap fuer die 2 staerksten Parameter (nach Varianz der Spalten).
    num_params = [
        k for k, v in spec.items()
        if k in full_results.columns and pd.api.types.is_numeric_dtype(full_results[k])
    ]
    if len(num_params) >= 2:
        variances = [(k, full_results[k].nunique()) for k in num_params]
        variances.sort(key=lambda x: -x[1])
        k1, k2 = variances[0][0], variances[1][0]
        plot_heatmap_top2(full_results, (k1, k2), out_dir / f"plot_heatmap_{k1}_vs_{k2}.png")

    # Stability box plot for top-K.
    top_k = full_results.nlargest(5, "objective_score")
    per_path_list = []
    labels = []
    stab_n = min(eval_paths_final // 2 if eval_paths_final >= 20 else 20, n_paths)
    for _, r in top_k.iterrows():
        params = {k: r[k] for k in spec.keys() if k in r}
        profits = []
        for i in range(stab_n):
            mids = {p: mc_paths[p][i] for p in products}
            rr = run_backtest_on_series(
                trader_cls, params, mids_by_product=mids,
                historical_depths=None, ordered_keys=None,
                position_limits=position_limits,
                max_ticks=max_ticks,
            )
            profits.append(rr["final_pnl"])
        per_path_list.append(np.array(profits))
        labels.append("|".join(f"{k}={params[k]}" for k in list(params)[:2]))
    plot_stability(per_path_list, labels, out_dir / "plot_stability_topK.png")

    print(f"[INFO] DONE in {time.time()-t0:.1f}s. Fill mode used: {FILL_MODE}.")
    print(f"[INFO] Artifacts written to: {out_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prosperity-4 MC/Grid Backtester (round-agnostic)")
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out-dir", type=Path, default=Path("./bt_out"))
    ap.add_argument("--trader", type=Path, default=None, help="Pfad zu externer trader.py")
    ap.add_argument("--round", dest="round_filter", type=int, default=None,
                    help="Optionaler Filter auf eine spezifische Round-Nummer (1,2,3,4,5).")
    ap.add_argument("--n-paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--risk-lambda", type=float, default=0.5,
                    help="Risk-Gewichtung in objective_score = mean - lambda*var")
    ap.add_argument("--max-ticks", type=int, default=2000,
                    help="Begrenze Ticks pro Backtest-Lauf (0 = kein Limit).")
    ap.add_argument("--exact-days", type=int, default=1,
                    help="Anzahl Tage, die im EXACT-Modus verwendet werden (0 = kein EXACT).")
    ap.add_argument("--eval-paths", type=int, default=50,
                    help="MC-Pfade pro Grid-Kombination (Speed/Varianz-Tradeoff).")
    ap.add_argument("--eval-paths-final", type=int, default=200,
                    help="MC-Pfade fuer finale Best-Param-Auswertung und Fan-Chart.")
    ap.add_argument("--grid-with-hist", action="store_true",
                    help="Fuehre den historischen EXACT-Lauf bei JEDER Grid-Kombi aus (langsam).")
    ap.add_argument("--ctf", action="store_true",
                    help="Aktiviere Two-Stage Coarse-to-Fine Grid Search.")
    ap.add_argument("--ctf-eval-coarse", type=int, default=10,
                    help="MC-Pfade pro Kombi in der coarse Stage (Default 10).")
    ap.add_argument("--ctf-eval-fine", type=int, default=100,
                    help="MC-Pfade pro Kombi in der fine Stage (Default 100).")
    ap.add_argument("--ctf-top-frac", type=float, default=0.10,
                    help="Top-Anteil der coarse Stage fuer die Verfeinerung (Default 0.10).")
    ap.add_argument("--ctf-n-interp", type=int, default=2,
                    help="Zwischenpunkte zwischen numerischen Top-Werten in der fine Stage (Default 2).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            trader_path=args.trader,
            n_paths=args.n_paths,
            seed=args.seed,
            risk_lambda=args.risk_lambda,
            round_filter=args.round_filter,
            max_ticks=(args.max_ticks if args.max_ticks and args.max_ticks > 0 else None),
            exact_days=args.exact_days,
            eval_paths=args.eval_paths,
            eval_paths_final=args.eval_paths_final,
            skip_hist_in_grid=(not args.grid_with_hist),
            ctf_enabled=args.ctf,
            ctf_eval_coarse=args.ctf_eval_coarse,
            ctf_eval_fine=args.ctf_eval_fine,
            ctf_top_frac=args.ctf_top_frac,
            ctf_n_interp=args.ctf_n_interp,
        )
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
