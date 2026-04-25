#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
round3_backtest_montecarlo.py
==============================

Prosperity-4 round-agnostic backtesting + Monte-Carlo + grid-search framework.

Sources of truth:
- P4 rules:                Prosperity.txt (in the workspace)
- P3 reference patterns:   README.md and FrankfurtHedgehogs_polished.txt
  (architecture / heuristics only -- NEVER treated as P4 facts)

Pipeline sections:
    1)  file discovery / parsing (official Prosperity CSVs, generic CSV fallback)
    2)  datamodel compatibility layer
        (local minimal Order / OrderDepth / TradingState if no `datamodel`)
    3)  order-book normalization + mid / micro price
    4)  Prosperity-compatible fill engine
        (EXACT + APPROX, sign convention, aggregated side limit check)
    5)  trader wrapper
        (loads external Trader by file path; otherwise uses DefaultMarketMaker)
    6)  feature engineering + diagnostics
    7)  Monte-Carlo path generation
        (block bootstrap -> residual bootstrap -> jump-aware perturbation)
    8)  parameter registry + single-stage / two-stage coarse-to-fine grid search
    9)  metrics (profit, drawdown, VaR, CVaR, turnover, stability)
    10) Pareto frontier + robust top-2 selection via neighborhood stability
    11) visualization
    12) artifact export

Usage:
    python round3_backtest_montecarlo.py --data-dir ./data --out-dir ./bt_out \
        [--trader path/to/trader.py] [--n-paths 1000] [--seed 42]

Runtime dependencies: numpy, pandas, matplotlib. Python 3.12 compatible.

SIGN CONVENTION (P4, authoritative from Prosperity.txt):
    - OrderDepth.sell_orders values are NEGATIVE volumes.
    - Order.quantity > 0 = buy,  Order.quantity < 0 = sell.
    - If aggregated buys OR aggregated sells of one product side would breach
      the absolute position limit, that entire side is dropped for the product.
    - traderData is the only persistence between iterations.
    - Unfilled remainders live as own resting quotes until end of iteration.
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

try:
    from scipy.stats import norm as _scipy_norm  # type: ignore
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover -- scipy is a runtime dep of pandas/sklearn
    _HAS_SCIPY = False
    _scipy_norm = None  # type: ignore

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Optional parallel-execution dependencies. The engine stays fully usable
# without them (single-threaded fallback) but `pip install joblib tqdm`
# unlocks process-pool parallelism with a live progress bar.
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    Parallel = None  # type: ignore
    delayed = None  # type: ignore

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# =============================================================================
# 0. GLOBAL CONFIG / ASSUMPTIONS
# =============================================================================

# ASSUMPTION: product list, limits and fields are not derivable from the
# space files alone. The framework stays product-agnostic; products and
# limits come from the loaded data (or a trader's LIMIT attribute, or an
# optional CLI override). No P3 product name is treated as a P4 fact.
DEFAULT_POSITION_LIMIT = 50  # ASSUMPTION: fallback only when data yields no limit.

# Known position limits from Prosperity.txt (P4 authoritative) and
# FrankfurtHedgehogs_polished.py (P3 Place-2 reference, 2024 data). These are
# used when the CSV files belong to those products -- inferring from observed
# order-book volume drastically over-estimates limits for small-cap products.
KNOWN_POSITION_LIMITS: Dict[str, int] = {
    # --- Prosperity 4 Round 1 (authoritative from Prosperity.txt) ---
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
    # --- Prosperity 4 Round 3 (team analysis in BRIEF_EQUIPE_ROUND3.md,
    # consistent with the P3 voucher structure -- VELVETFRUIT_EXTRACT plays
    # the role of VOLCANIC_ROCK, VEV_* are the call vouchers). These are
    # working values; if the official P4 limits differ, override via
    # --position-limit PRODUCT=N. ASSUMPTION so long as Prosperity.txt does
    # not yet list the Round 3 limits. ---
    "VELVETFRUIT_EXTRACT": 400,
    "VEV_4000": 200,
    "VEV_4500": 200,
    "VEV_5000": 200,
    "VEV_5100": 200,
    "VEV_5200": 200,
    "VEV_5300": 200,
    "VEV_5400": 200,
    "VEV_5500": 200,
    "VEV_6000": 200,
    "VEV_6500": 200,
    "HYDROGEL_PACK": 75,  # ASSUMPTION: no official reference yet; sized
                          # like MAGNIFICENT_MACARONS (P3 auxiliary product).
    # --- Prosperity 3 2024 (from FrankfurtHedgehogs reference) ---
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75,
}

# ASSUMPTION: official Prosperity CSVs use a 100-ms tick grid (timestamp in
# steps of 100). The framework only relies on numeric ordering.
TICK_STEP = 100

# Fill-mode flag: set at runtime. Promoted to "EXACT" when L1/L2 order books
# are available for the current backtest run.
FILL_MODE = "APPROX"

# Default simulated bid-ask spread (in ticks) used in APPROX-fill mode when no
# per-product override is provided. Historically this was hardcoded to 2.0
# (bid = floor(mid - 1), ask = ceil(mid + 1)); for products that genuinely
# trade at much wider spreads (e.g. VEV vouchers ~ 4-12 ticks) this distorts
# fill economics. Override per product via :func:`compute_mean_historical_spread`
# or the ``--sim-spread`` CLI flag.
DEFAULT_SIM_SPREAD = 2.0


# =============================================================================
# 1. DATAMODEL COMPATIBILITY LAYER
# =============================================================================
# Minimal local versions that are compatible with the official Prosperity
# `datamodel`. If that module is importable (e.g. inside the IMC sandbox),
# it is preferred over the local fallbacks.

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
            # buy_orders:  price -> positive volume
            # sell_orders: price -> NEGATIVE volume (P4 sign convention)
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

# Match arbitrary rounds and arbitrary day suffixes (e.g. day_0, day_-1, day_-1-6).
# The first integer after 'day' is the DAY; any additional trailing integers
# are treated as file/upload version suffixes.
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
    """Discover Prosperity price / trade CSVs plus generic fallback CSVs.

    If ``round_filter`` is set, only files of that round are returned.
    ``data_dir`` may be a single file, a directory, or non-existent."""
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
    """Prosperity CSVs use ';' as delimiter; fall back to ','."""
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:  # wrong delimiter
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep=",")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_prices(paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate price snapshots. Standard columns (official format):
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
            # group(2) = day, group(1) = round.
            df["day"] = int(m.group(2)) if m else 0
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    # ask_volume_x is positive in the CSVs; we convert to negative sell_orders
    # later during order-book normalization.
    # --- Robustness: drop rows with non-positive mid_price (book outage -> 0).
    # These rows poison log-returns downstream (log(0) -> -inf, pct_change -> inf)
    # and make feature_diagnostics/monte-carlo summaries come back as NaN.
    if "mid_price" in out.columns:
        bad = (~np.isfinite(out["mid_price"])) | (out["mid_price"] <= 0)
        if bad.any():
            print(f"[WARN] dropping {int(bad.sum())} price rows with mid_price<=0 or NaN")
            out = out.loc[~bad].reset_index(drop=True)
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
    """Convert price columns into OrderDepth objects per (day, timestamp)."""
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
                # Prosperity sign convention: sell_orders are negative.
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
# Rules (Prosperity.txt):
# - Orders match immediately against resting quotes at or better than the order's price.
# - Execution happens at the price of the resting order.
# - Unfilled remainders live as own resting quotes until end of iteration and
#   may be hit by bot flow (APPROX mode models this conservatively via a
#   next-tick mid-crossing heuristic).
# - Aggregated side limit check: if aggregated buys OR aggregated sells of a
#   product would breach the absolute position limit, that side is discarded
#   entirely for this product.


@dataclass
class FillResult:
    product: str
    price: int
    quantity: int  # signed: +buy, -sell (from the trader's perspective)
    timestamp: int


def _aggregate_side_limit_check(
    orders: List[Order], current_position: int, position_limit: int
) -> List[Order]:
    """Enforce the Prosperity rule: if aggregated buys (or aggregated sells) of
    one side would breach the absolute position limit, that entire side of the
    product is discarded."""
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    buy_sum = sum(o.quantity for o in buys)
    sell_sum = sum(o.quantity for o in sells)  # negative
    # After aggregation we require: current_position + buy_sum  <= +limit
    #                         and: current_position + sell_sum >= -limit.
    kept: List[Order] = []
    if buys:
        if current_position + buy_sum <= position_limit:
            kept.extend(buys)
        # else: entire buy side dropped.
    if sells:
        if current_position + sell_sum >= -position_limit:
            kept.extend(sells)
        # else: entire sell side dropped.
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
    """Run full matching for one tick.

    Returns
    -------
    fills : list of FillResult (signed quantity from the trader's perspective)
    position_delta : per-product signed delta
    """
    # Fast path: the vast majority of ticks in a mean-reverter or market-
    # making trader return no orders. Avoid all per-tick setup in that case.
    if not orders_by_product:
        return [], {}

    all_fills: List[FillResult] = []
    delta: Dict[str, int] = defaultdict(int)

    for prod, orders in orders_by_product.items():
        if not orders:
            continue
        pos_now = position.get(prod, 0)
        limit = position_limits.get(prod, DEFAULT_POSITION_LIMIT)

        # 1) Side-aggregated limit check (P4 rule).
        orders = _aggregate_side_limit_check(orders, pos_now, limit)
        if not orders:
            continue

        depth = depths.get(prod)
        if depth is None:
            continue

        # Local copies of the opposite sides so we can deplete volumes.
        sell_book = dict(depth.sell_orders)  # px -> negative
        buy_book = dict(depth.buy_orders)    # px -> positive

        working_pos = pos_now
        # Sort so buy orders hit lowest asks first, sell orders hit highest
        # bids first (priority by aggressiveness).
        orders_sorted = sorted(
            orders, key=lambda o: (-o.price if o.quantity > 0 else o.price)
        )
        unfilled: List[Order] = []

        for o in orders_sorted:
            qty_left = o.quantity
            if qty_left > 0:
                # BUY: match against sell_book with px <= o.price, cheapest first.
                for ask_px in sorted(sell_book.keys()):
                    if ask_px > o.price or qty_left <= 0:
                        break
                    avail = -sell_book[ask_px]  # positive
                    # Hard limit: working_pos + fill <= limit.
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
                # SELL: match against buy_book with px >= o.price, highest first.
                for bid_px in sorted(buy_book.keys(), reverse=True):
                    if bid_px < o.price or qty_left >= 0:
                        break
                    avail = buy_book[bid_px]  # positive
                    # Hard limit: sell -> new position = working_pos - take >= -limit.
                    max_by_limit = limit + working_pos
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

            # Remainders live on as resting quotes.
            if qty_left != 0:
                unfilled.append(Order(o.symbol, o.price, qty_left))

        # 2) Unfilled remainders: in APPROX mode we model bot flow via a
        # "next-tick mid crossing" heuristic. EXACT mode would require a
        # complete bot-flow model.
        nm = next_mid.get(prod)
        if nm is not None and unfilled and fill_mode == "APPROX":
            for o in unfilled:
                if o.quantity > 0 and nm < o.price - 0.5:
                    # Our bid is accepted by the market -> fill at o.price.
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
    Generic parametrized market-making trader.
    Prosperity-compatible: stateless, persistence only via ``traderData``.
    Used as the default when no external trader is supplied.
    """

    # Parameter hook the grid-search engine keys off of.
    # mathematical_low: floor for fair value. Never -inf; -1e6 effectively
    # means "no floor" for typical Prosperity price levels.
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
        return s[:49_000]  # Prosperity soft limit ~ 50k.

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

            # Clip via mathematical_low (floor). Only applies if floor > -inf.
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
    """Load an external Trader class from a Python file via importlib.

    Supports filenames with hyphens/dots because we load by path rather than
    module name. Expects ``class Trader:`` with ``run(self, state)``.
    Returns None on failure -- caller falls back to DefaultMarketMaker.

    Also injects local datamodel shims under the module name ``datamodel``
    into ``sys.modules`` before import, so traders that ``from datamodel
    import Order, OrderDepth, TradingState`` work even outside the IMC sandbox.
    """
    if not path or not path.exists():
        return None

    # 1) Make `from datamodel import ...` work for traders loaded in this process.
    if "datamodel" not in sys.modules:
        import types as _types
        dm = _types.ModuleType("datamodel")
        dm.Order = Order
        dm.OrderDepth = OrderDepth
        dm.TradingState = TradingState
        dm.Trade = Trade
        dm.Observation = Observation
        sys.modules["datamodel"] = dm

    # 2) Load by path (handles hyphens, weird names, same-dir collisions).
    import importlib.util as _ilu
    mod_name = f"_trader_{re.sub(r'[^0-9A-Za-z_]', '_', path.stem)}_{abs(hash(str(path))) % 10**6}"
    try:
        spec = _ilu.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            print(f"[WARN] Could not build importlib spec for {path}")
            return None
        mod = _ilu.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[WARN] Failed to import external trader {path}: {e}")
        return None

    TraderCls = getattr(mod, "Trader", None)
    if TraderCls is None:
        print(f"[WARN] {path} does not define a class named 'Trader'.")
        return None
    return TraderCls


# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================


def compute_mean_historical_spread(
    prices_df: pd.DataFrame,
    fallback: float = DEFAULT_SIM_SPREAD,
    min_spread: float = 1.0,
) -> Dict[str, float]:
    """Per-product mean of (ask_price_1 - bid_price_1) across the dataset.

    Used to seed the simulated bid-ask spread of the APPROX-fill backtester.
    Products without usable L1 quotes fall back to ``fallback`` (default 2.0).
    Spreads below ``min_spread`` are clipped up so the simulated bid is always
    strictly below the simulated ask after rounding.
    """
    out: Dict[str, float] = {}
    if prices_df.empty:
        return out
    if not {"bid_price_1", "ask_price_1", "product"}.issubset(prices_df.columns):
        return out
    spread = (prices_df["ask_price_1"] - prices_df["bid_price_1"]).astype(float)
    spread = spread.where(spread > 0)  # drop crossed/missing rows
    grouped = spread.groupby(prices_df["product"]).mean()
    for prod, val in grouped.items():
        if pd.isna(val):
            out[str(prod)] = float(fallback)
        else:
            out[str(prod)] = float(max(min_spread, val))
    return out


def build_feature_frame(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Basic features per (day, timestamp, product)."""
    if prices_df.empty:
        return pd.DataFrame()
    df = prices_df.copy()
    if "mid_price" not in df.columns:
        # Reconstruct from bid_price_1 / ask_price_1.
        bp = df.get("bid_price_1")
        ap = df.get("ask_price_1")
        if bp is not None and ap is not None:
            df["mid_price"] = (bp + ap) / 2
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    # Mask non-positive mid so log-returns are NaN instead of -inf; downstream
    # consumers drop NaN safely.
    safe_mid = df["mid_price"].where(df["mid_price"] > 0)
    df["log_ret"] = (
        safe_mid.groupby(df["product"]).apply(lambda s: np.log(s).diff()).reset_index(level=0, drop=True)
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
        rets = grp["log_ret"].replace([np.inf, -np.inf], np.nan).dropna()
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


def _block_bootstrap_returns(
    rets: np.ndarray, n: int, block_len: int, rng: np.random.Generator
) -> np.ndarray:
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
    """Generate ``(n_paths, T)`` mid-price matrices per product.
    Generator order: block bootstrap -> residual bootstrap -> jump-aware perturbation.
    For coupled products we sample via shared time indices so cross-asset
    correlations are preserved.
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
    # Drop products that are entirely empty.
    pv = pv.dropna(axis=1, how="all")
    if pv.empty:
        return {}
    # Safety net: fill any remaining NaN with the column mean.
    pv = pv.fillna(pv.mean(numeric_only=True))
    products = list(pv.columns)
    T = len(pv)
    if T < 3:
        return {p: np.tile(pv[p].values, (n_paths, 1)) for p in products}

    # Gemeinsame Bootstrap-Indizes, um Cross-Asset-Korrelationen zu erhalten.
    rets = np.log(pv.values[1:] / pv.values[:-1])  # shape (T-1, P)
    out: Dict[str, np.ndarray] = {p: np.empty((n_paths, T)) for p in products}

    for i in range(n_paths):
        # Block-bootstrap indices across the TIME axis -> cross-asset correlation preserved.
        sampled_idx = np.empty(T - 1, dtype=np.int64)
        filled = 0
        while filled < T - 1:
            start = rng.integers(0, max(1, len(rets) - block_len + 1))
            take = min(block_len, T - 1 - filled)
            sampled_idx[filled : filled + take] = np.arange(start, start + take)
            filled += take
        sampled = rets[sampled_idx]  # (T-1, P)

        # Mix in ~30% residual bootstrap for extra path diversity.
        if rng.random() < 0.3:
            for k in range(sampled.shape[1]):
                sampled[:, k] = _residual_bootstrap_returns(rets[:, k], len(sampled), rng)

        # Jump perturbation.
        for k, prod in enumerate(products):
            std_k = rets[:, k].std() if rets.shape[0] > 1 else 0.0
            jump_frac = float((np.abs(rets[:, k]) > 4 * std_k).mean()) if std_k > 0 else 0.0
            sampled[:, k] = _add_jumps(sampled[:, k], rng, jump_frac, 3 * std_k)

        # Re-integrate into prices, anchored at the historical start price.
        log_p0 = np.log(pv.values[0])
        log_path = np.vstack([log_p0, log_p0 + np.cumsum(sampled, axis=0)])
        prices = np.exp(log_path)
        for k, prod in enumerate(products):
            out[prod][i] = prices[:, k]

    return out


# -----------------------------------------------------------------------------
# 7b. ORNSTEIN-UHLENBECK PATH GENERATION + CALIBRATION DIAGNOSTICS
# -----------------------------------------------------------------------------
#
# The block-bootstrap above reshuffles historical returns, which preserves
# empirical marginal distributions but assumes no parametric structure. For
# products that mean-revert around a long-run level (a spread, a
# calibrated fair value, etc.) an Ornstein-Uhlenbeck model is a closer fit:
#
#     dS_t = theta * (mu - S_t) * dt + sigma * dW_t
#
# Discretised at unit dt the exact transition is
#
#     S_{t+1} = mu + (S_t - mu) * exp(-theta) + sigma_eff * eps,
#     sigma_eff = sigma * sqrt((1 - exp(-2 theta)) / (2 theta))
#
# Calibration is AR(1) regression on lagged levels: regress (S_{t+1} - S_t)
# on (S_t - mean); the slope is theta, residual std scales to sigma, and the
# intercept pins mu. This is the maximum-likelihood estimator under Gaussian
# shocks (Aït-Sahalia, 2002) and an order of magnitude faster than MLE for
# our grid sizes.
#
# Validation diagnostics returned per product:
#   - theta, mu, sigma (estimated)
#   - half_life = log(2) / theta  (intuitive: ticks to close half the gap)
#   - r2_ar1   (how well the AR(1) form actually fits)
#   - adf_like (unit-root-ish statistic: theta / se(theta); >2 ~ reverting)
#   - resid_std, resid_skew, resid_kurt  (Gaussianity check)
#   - ljung_box_lag5_p  (whitening check on residuals)
#   - mean_reversion_acf_decay  (1-step AC of centred level; should be < 1)
#
# The user can OVERRIDE any of theta / mu / sigma via CLI; unfurnished values
# fall back to the calibrated estimate. A per-product override dict is
# accepted as well (useful when Prosperity.txt gives a fair value for one
# product but not others).


def _ar1_fit(series: np.ndarray) -> Dict[str, float]:
    """Fit an AR(1) mean-reverting process to a 1D level series and return
    diagnostics alongside ``theta / mu / sigma``. Designed to be robust to
    short / flat / NaN-containing input: returns safe defaults rather than
    raising, so an unfittable product silently falls back to historical
    mean + zero vol and will produce flat paths.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    base = dict(theta=0.0, mu=float(x.mean()) if n else 0.0, sigma=0.0,
                half_life=float("inf"), r2_ar1=0.0, adf_like=0.0,
                resid_std=0.0, resid_skew=0.0, resid_kurt=0.0,
                ljung_box_lag5_p=float("nan"), mean_reversion_acf_decay=1.0,
                n_obs=int(n))
    if n < 10:
        return base
    # Difference regression: dx_t = alpha + beta * x_t + eps
    x_lag = x[:-1]
    dx = np.diff(x)
    xm = x_lag.mean()
    xl_c = x_lag - xm
    denom = float(np.sum(xl_c ** 2))
    if denom <= 0:
        return base
    beta = float(np.sum(xl_c * (dx - dx.mean())) / denom)
    alpha = float(dx.mean() - beta * xm)
    resid = dx - (alpha + beta * x_lag)
    # theta from the continuous-time mapping beta = -(1 - exp(-theta)).
    # Numerical guard: clip beta to (-1 + 1e-9, 0) before inverting.
    beta_c = max(min(beta, -1e-9), -1 + 1e-9) if beta < 0 else 0.0
    theta = -math.log(1 + beta_c) if beta_c < 0 else 0.0
    mu = (-alpha / beta) if beta_c < 0 and beta != 0 else float(x.mean())
    # Long-run sigma from sigma_eff (residual std of dx) via
    # sigma_eff^2 = sigma^2 * (1 - exp(-2 theta)) / (2 theta).
    sigma_eff = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0
    if theta > 1e-6:
        factor = (1.0 - math.exp(-2.0 * theta)) / (2.0 * theta)
        sigma = sigma_eff / math.sqrt(factor) if factor > 0 else sigma_eff
    else:
        sigma = sigma_eff  # driftless random walk limit
    # Diagnostics.
    ss_tot = float(np.sum((dx - dx.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    # Standard error of beta; adf_like = -beta / se(beta). Large positive
    # value means strong mean-reversion (mirrors ADF t-stat in spirit).
    se_beta = math.sqrt(float(np.sum(resid ** 2)) / max(n - 2, 1) / denom)
    adf_like = (-beta / se_beta) if se_beta > 0 else 0.0
    # Residual higher moments.
    rs = resid - resid.mean()
    rsd = float(np.std(rs))
    if rsd > 0:
        skew = float(np.mean((rs / rsd) ** 3))
        kurt = float(np.mean((rs / rsd) ** 4) - 3.0)
    else:
        skew = kurt = 0.0
    # Ljung-Box at lag 5 on residuals (whitening check). Small p -> structure
    # left over -> AR(1) is an incomplete fit.
    try:
        lags = 5
        n_r = len(rs)
        acfs = []
        var_r = float(np.var(rs))
        for k in range(1, lags + 1):
            if n_r - k <= 0 or var_r <= 0:
                break
            ck = float(np.mean(rs[:-k] * rs[k:])) / var_r
            acfs.append(ck)
        if acfs:
            q = n_r * (n_r + 2) * sum((a ** 2) / (n_r - k - 1)
                                       for k, a in enumerate(acfs))
            # Approximate chi-square tail probability via survival of gamma.
            from math import gamma as _gamma
            k_df = len(acfs)
            # Use an upper-tail approximation (Wilson-Hilferty).
            t = (q / k_df) ** (1 / 3)
            m = 1 - 2 / (9 * k_df)
            s = math.sqrt(2 / (9 * k_df))
            z = (t - m) / s
            lb_p = 0.5 * math.erfc(z / math.sqrt(2))
        else:
            lb_p = float("nan")
    except Exception:
        lb_p = float("nan")
    # 1-lag ACF of centred level (should be < 1 for reverting series).
    xc = x - x.mean()
    if n > 2 and float(np.var(xc)) > 0:
        acf1 = float(np.mean(xc[:-1] * xc[1:])) / float(np.var(xc))
    else:
        acf1 = 1.0
    half_life = (math.log(2) / theta) if theta > 1e-9 else float("inf")
    return dict(theta=float(theta), mu=float(mu), sigma=float(sigma),
                half_life=float(half_life), r2_ar1=float(r2),
                adf_like=float(adf_like), resid_std=float(rsd),
                resid_skew=float(skew), resid_kurt=float(kurt),
                ljung_box_lag5_p=float(lb_p),
                mean_reversion_acf_decay=float(acf1), n_obs=int(n))


def calibrate_ou(
    prices_df: pd.DataFrame,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit OU per product. ``overrides`` may supply user-specified
    ``theta`` / ``mu`` / ``sigma`` per product; any key present wins over
    the calibrated value. Returns a dict keyed by product with the fit
    parameters plus all diagnostics (see :func:`_ar1_fit`).
    """
    overrides = overrides or {}
    global_ov = overrides.get("__GLOBAL__", {})
    out: Dict[str, Dict[str, float]] = {}
    for prod, sub in prices_df.groupby("product"):
        fit = _ar1_fit(sub["mid_price"].values)
        # Global overrides apply first, per-product overrides win over those.
        ov: Dict[str, float] = dict(global_ov)
        ov.update(overrides.get(prod, {}))
        # User overrides take precedence but leave diagnostics intact so
        # the user can see how far their override is from the data.
        for k in ("theta", "mu", "sigma"):
            if k in ov and ov[k] is not None:
                fit[f"{k}_calibrated"] = fit[k]
                fit[k] = float(ov[k])
        # Derived half-life reflects the effective theta actually used.
        fit["half_life"] = (math.log(2) / fit["theta"]) if fit["theta"] > 1e-9 else float("inf")
        out[prod] = fit
    return out


def generate_monte_carlo_paths_ou(
    prices_df: pd.DataFrame,
    n_paths: int = 1000,
    seed: int = 42,
    params: Optional[Dict[str, Dict[str, float]]] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
    horizon: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Generate mean-reverting paths from a calibrated OU process.

    Returns ``(paths_by_product, calibration_diagnostics)``.

    Shocks are drawn independently per product (no cross-asset coupling).
    If that matters for your trader, stick with the bootstrap generator --
    OU here is single-asset-parametric by design. Path length defaults to
    the per-product historical length unless ``horizon`` is set.
    """
    if prices_df.empty:
        return {}, {}
    rng = np.random.default_rng(seed)
    if params is None:
        params = calibrate_ou(prices_df, overrides=overrides)
    out: Dict[str, np.ndarray] = {}
    for prod, fit in params.items():
        sub = prices_df[prices_df["product"] == prod].sort_values(["day", "timestamp"])
        levels = sub["mid_price"].values
        levels = levels[np.isfinite(levels)]
        T = int(horizon) if horizon is not None else int(len(levels))
        if T <= 1 or len(levels) == 0:
            out[prod] = np.zeros((n_paths, max(T, 1)))
            continue
        theta = float(fit["theta"])
        mu = float(fit["mu"])
        sigma = float(fit["sigma"])
        # Exact transition moments (unit dt).
        decay = math.exp(-theta) if theta > 0 else 1.0
        if theta > 1e-9:
            sigma_step = sigma * math.sqrt((1.0 - math.exp(-2.0 * theta)) / (2.0 * theta))
        else:
            sigma_step = sigma  # degenerate random-walk limit
        arr = np.empty((n_paths, T), dtype=float)
        arr[:, 0] = float(levels[0])
        if sigma_step > 0 and T > 1:
            eps = rng.standard_normal(size=(n_paths, T - 1))
            # Vectorised recursion: S_{t+1} = mu + decay * (S_t - mu) + sigma_step * eps
            for t in range(1, T):
                arr[:, t] = mu + decay * (arr[:, t - 1] - mu) + sigma_step * eps[:, t - 1]
        else:
            arr[:, 1:] = arr[:, :1]
        out[prod] = arr
    return out, params


def plot_ou_calibration(
    prices_df: pd.DataFrame,
    mc_paths: Dict[str, np.ndarray],
    ou_params: Dict[str, Dict[str, float]],
    out: Path,
    n_overlay: int = 20,
) -> None:
    """3-panel validation plot PER product (stacked vertically):
      1. historical level with long-run mean + +/-2 sigma band
      2. overlay of ``n_overlay`` simulated paths on the same axes
      3. histogram of simulated terminal levels vs. historical mean/std
    Skips silently on error.

    Only products that actually got OU-simulated (mc_paths entry present AND
    theta > 1e-6) are plotted -- products on the bootstrap path are filtered
    out, so the figure does not ship 13 empty panels per run.
    """
    # Only plot products that actually got OU-simulated -- mc_paths is
    # already filtered by the caller to OU products only, so we key off
    # its presence (not ou_params, which keeps all products for diagnostics).
    products = [p for p in ou_params.keys()
                if p in mc_paths and mc_paths[p] is not None
                and mc_paths[p].size]
    if not products:
        print("[INFO] plot_ou_calibration: no OU-simulated products -- skipping")
        return
    rows = len(products)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.2 * rows), squeeze=False)
    for i, prod in enumerate(products):
        ax_hist, ax_sim, ax_dist = axes[i]
        sub = prices_df[prices_df["product"] == prod].sort_values(["day", "timestamp"])
        hist = sub["mid_price"].values.astype(float)
        hist = hist[np.isfinite(hist)]
        fit = ou_params[prod]
        mu = fit["mu"]
        # Long-run std of an OU process: sigma / sqrt(2*theta).
        if fit["theta"] > 1e-9:
            lr_std = fit["sigma"] / math.sqrt(2.0 * fit["theta"])
        else:
            lr_std = float(np.std(hist)) if len(hist) else 0.0
        # Panel 1: historical.
        ax_hist.plot(hist, lw=0.7, color="black")
        ax_hist.axhline(mu, color="red", lw=1.0, label=f"mu={mu:.2f}")
        ax_hist.axhline(mu + 2 * lr_std, color="red", lw=0.6, ls="--",
                        label=f"+/-2 sigma_lr ({lr_std:.2f})")
        ax_hist.axhline(mu - 2 * lr_std, color="red", lw=0.6, ls="--")
        ax_hist.set_title(f"{prod}: historical level")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(alpha=0.3)
        # Panel 2: simulated overlay.
        paths = mc_paths.get(prod)
        if paths is not None and paths.size:
            k = min(n_overlay, paths.shape[0])
            for j in range(k):
                ax_sim.plot(paths[j], lw=0.4, alpha=0.6)
            ax_sim.axhline(mu, color="red", lw=1.0)
            ax_sim.set_title(
                f"{prod}: {k} simulated paths  |  theta={fit['theta']:.4f}, "
                f"half_life={fit['half_life']:.1f} ticks"
            )
            ax_sim.grid(alpha=0.3)
            # Panel 3: terminal distribution.
            term = paths[:, -1]
            ax_dist.hist(term, bins=30, color="steelblue", alpha=0.8)
            ax_dist.axvline(mu, color="red", lw=1.0, label=f"mu={mu:.2f}")
            ax_dist.axvline(float(np.mean(hist)), color="black", lw=1.0, ls=":",
                            label=f"hist mean={float(np.mean(hist)):.2f}")
            ax_dist.set_title(f"{prod}: terminal level dist (n={len(term)})")
            ax_dist.legend(fontsize=8)
            ax_dist.grid(alpha=0.3)
    fig.suptitle("OU calibration & simulation diagnostics", y=1.0, fontsize=12)
    save_plot(fig, out)


# =============================================================================
# 8. PARAMETER REGISTRY + GRID SEARCH
# =============================================================================


def extract_param_spec(trader_cls) -> Dict[str, Any]:
    """Best-effort extraction of a PARAM_SPEC from an external Trader class.

    Detection order:
      1. Class attribute ``PARAM_SPEC`` (preferred; native format).
      2. Class attribute ``CONFIG`` (dict of tunable constants).
         ASSUMPTION: CONFIG values are numeric/bool; we build small grids
         around each default (+/- 1 step / +/-50% for floats).
      3. Otherwise: no parameters detected -> return ``{}``. The pipeline
         will still run a single evaluation against the fixed trader.
    """
    spec = getattr(trader_cls, "PARAM_SPEC", None)
    if isinstance(spec, dict) and spec:
        return spec
    cfg = getattr(trader_cls, "CONFIG", None)
    if isinstance(cfg, dict) and cfg:
        out = {}
        for k, v in cfg.items():
            if isinstance(v, bool):
                out[k] = {"type": "bool", "grid": [False, True]}
            elif isinstance(v, int):
                out[k] = {"type": "int", "grid": sorted(set([max(0, v - 1), v, v + 1]))}
            elif isinstance(v, float):
                out[k] = {"type": "float", "grid": sorted(set([v * 0.5, v, v * 1.5]))}
        return out
    return {}


def build_grid(spec: Dict[str, Any], focus_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    keys = focus_keys if focus_keys else list(spec.keys())
    if not keys:
        # Empty spec -> exactly one "combo" with no overrides so the pipeline
        # still performs a single evaluation of the underlying trader.
        return [{}]
    grids = [spec[k]["grid"] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]


def coarse_to_fine_grid(
    spec: Dict[str, Any], run_batch: Callable[[List[Dict[str, Any]]], pd.DataFrame], top_k: int = 6
) -> pd.DataFrame:
    """Legacy auto-coarse-to-fine (only triggers when >500 combos). Kept for
    backward compatibility; the explicit path is :func:`two_stage_ctf`."""
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
    """Refine a numeric value list by inserting ``n_interp`` equidistant points
    between neighbouring values. Bool / categorical lists are returned unchanged."""
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
    # Keep the refined grid integer-typed if all inputs were ints.
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
    rank_metric: str = "sharpe",
    min_trades: float = 5.0,
) -> pd.DataFrame:
    """Explicit two-stage coarse-to-fine grid search.

    Stage 1: evaluate the full ``spec`` with ``run_batch_coarse`` (low MC
             budget -- fast noisy scan of the whole parameter space).
    Stage 2: pick the top ``top_frac`` rows of stage 1, collect the per-key
             value sets that appear in the top region, refine numeric keys
             via :func:`_refine_numeric_grid`, take the cross product and
             evaluate with ``run_batch_fine`` (higher MC budget).

    Returns a concatenated DataFrame with a ``stage`` column in
    ``{"coarse", "fine"}``.
    """
    coarse_combos = build_grid(spec)
    print(f"[CTF] stage-1 coarse grid: {len(coarse_combos)} combos")
    if not spec:
        # Nothing to tune -- single evaluation already covered by coarse run.
        coarse_df = run_batch_coarse(coarse_combos)
        return coarse_df.assign(stage="coarse")
    coarse_df = run_batch_coarse(coarse_combos)
    if coarse_df.empty:
        return coarse_df.assign(stage="coarse")
    coarse_df = coarse_df.copy()
    coarse_df["stage"] = "coarse"

    # --- Selection pool: drop zero-/low-trade configs BEFORE ranking so the
    # top region reflects strategies that actually engage the market, not
    # degenerate no-op configs whose `objective_score == 0`.
    pool = coarse_df
    if "mean_trades_per_path" in pool.columns:
        active = pool[pool["mean_trades_per_path"] >= min_trades]
        if len(active) >= max(min_top, 4):
            pool = active
            print(f"[CTF] filtered to {len(pool)}/{len(coarse_df)} active configs "
                  f"(mean_trades_per_path >= {min_trades})")
        else:
            print(f"[CTF] WARN: only {len(active)} active configs, keeping full pool")
    # Ranking metric: default "sharpe" rewards consistency; fall back cleanly
    # if the requested column is missing.
    if rank_metric not in pool.columns:
        print(f"[CTF] WARN: rank_metric='{rank_metric}' missing, falling back to 'mean_profit'")
        rank_metric = "mean_profit" if "mean_profit" in pool.columns else "objective_score"
    # Protect against NaN/inf values.
    pool = pool[np.isfinite(pool[rank_metric])]
    n_top = int(max(min_top, min(max_top, math.ceil(len(pool) * top_frac))))
    top = pool.nlargest(n_top, rank_metric)
    print(f"[CTF] stage-1 kept top {len(top)} rows (~{top_frac*100:.0f}%) "
          f"ranked by '{rank_metric}' for refinement")

    # Build fine_spec: per key take the values that occur in the top region,
    # plus optional numeric interpolation.
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
    # Dedupe against combos already evaluated in the coarse stage
    # (exact parameter-value match).
    coarse_keys = set(
        tuple(sorted((k, coarse_df.iloc[i][k]) for k in spec.keys() if k in coarse_df.columns))
        for i in range(len(coarse_df))
    )
    fine_unique = []
    for c in fine_combos:
        key = tuple(sorted(c.items()))
        if key not in coarse_keys:
            fine_unique.append(c)
    print(f"[CTF] stage-2 fine grid: {len(fine_unique)} new combos (of {len(fine_combos)} generated)")

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


def infer_position_limits(
    prices_df: pd.DataFrame,
    overrides: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Resolve per-product position limits.

    Priority: user CLI overrides -> KNOWN_POSITION_LIMITS (P4 authoritative /
    P3 reference) -> DEFAULT_POSITION_LIMIT fallback. The previous
    ``max(observed volume)`` heuristic was removed because it over-estimated
    small-cap limits by 3-5x (e.g. JAMS reported 350 official vs observed
    volume peak of 305; CROISSANTS official 250 vs observed 172) and under-
    estimated none -- no realistic P4 product has a higher limit than
    suggested by the order book. Using authoritative limits is a correctness
    win for every metric that depends on capped position (PnL, drawdown,
    turnover, hit_rate_active).
    """
    overrides = overrides or {}
    if prices_df.empty:
        return dict(overrides)
    limits: Dict[str, int] = {}
    for prod in prices_df["product"].unique():
        prod_s = str(prod)
        if prod_s in overrides:
            limits[prod_s] = int(overrides[prod_s])
        elif prod_s in KNOWN_POSITION_LIMITS:
            limits[prod_s] = int(KNOWN_POSITION_LIMITS[prod_s])
        else:
            limits[prod_s] = int(DEFAULT_POSITION_LIMIT)
            print(f"[WARN] no known position limit for '{prod_s}', "
                  f"falling back to DEFAULT_POSITION_LIMIT={DEFAULT_POSITION_LIMIT}. "
                  f"Pass --position-limit {prod_s}=N to override.")
    return limits


def build_market_trades_index(
    trades_df: pd.DataFrame,
) -> Dict[Tuple[int, int], Dict[str, List["Trade"]]]:
    """Group recorded market trades by (day, timestamp) -> product -> List[Trade].

    Prosperity trade CSVs typically contain the columns ``timestamp``,
    ``symbol`` (or ``product``), ``price``, ``quantity``, ``buyer``,
    ``seller``, and optionally ``day``. Unknown columns are ignored.
    Returns an empty dict if the frame is empty or lacks required fields.
    """
    index: Dict[Tuple[int, int], Dict[str, List[Trade]]] = {}
    if trades_df is None or trades_df.empty:
        return index
    cols = set(trades_df.columns)
    sym_col = "symbol" if "symbol" in cols else ("product" if "product" in cols else None)
    if sym_col is None or "timestamp" not in cols:
        return index
    for row in trades_df.itertuples(index=False):
        d = dict(zip(trades_df.columns, row))
        try:
            day = int(d.get("day", 0))
            ts = int(d.get("timestamp", 0))
            sym = str(d.get(sym_col))
            price = float(d.get("price", 0))
            qty = int(float(d.get("quantity", 0)))
        except Exception:
            continue
        if not sym or sym == "nan":
            continue
        buyer = str(d.get("buyer", "") or "")
        seller = str(d.get("seller", "") or "")
        tr = Trade(
            symbol=sym,
            price=int(round(price)) if float(price).is_integer() else price,  # type: ignore[arg-type]
            quantity=qty,
            buyer=buyer,
            seller=seller,
            timestamp=ts,
        )
        index.setdefault((day, ts), {}).setdefault(sym, []).append(tr)
    return index


def extract_trader_limits(trader_cls) -> Dict[str, int]:
    """Pull hard position limits from an external Trader class if present.

    Recognized attributes (first match wins): ``LIMIT``, ``LIMITS``,
    ``POSITION_LIMIT``, ``POSITION_LIMITS``. Value must be a ``dict`` mapping
    product symbol -> integer limit. Returns ``{}`` if nothing is declared.
    """
    for attr in ("LIMIT", "LIMITS", "POSITION_LIMIT", "POSITION_LIMITS"):
        val = getattr(trader_cls, attr, None)
        if isinstance(val, dict) and val:
            try:
                return {str(k): int(v) for k, v in val.items()}
            except Exception:
                continue
    return {}


def run_backtest_on_series(
    trader_cls,
    params: Dict[str, Any],
    mids_by_product: Dict[str, np.ndarray],
    historical_depths: Optional[Dict[Tuple[int, int], Dict[str, OrderDepth]]],
    ordered_keys: Optional[List[Tuple[int, int]]],
    position_limits: Dict[str, int],
    max_ticks: Optional[int] = None,
    market_trades_index: Optional[Dict[Tuple[int, int], Dict[str, List["Trade"]]]] = None,
    sim_spread_by_product: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run a single backtest pass.

    If ``historical_depths`` and ``ordered_keys`` are provided the run uses
    EXACT-fill mode against the real recorded order books. Otherwise it falls
    back to APPROX-fill mode against the per-product mid-price series, with a
    synthetic order book centred on the mid. The simulated half-spread is
    ``sim_spread_by_product[product] / 2`` (default :data:`DEFAULT_SIM_SPREAD`)
    so a value of 2.0 reproduces the legacy ``floor(mid - 1) / ceil(mid + 1)``
    behaviour while wider products (e.g. VEV vouchers) can use their measured
    historical spread instead.

    If ``market_trades_index`` is provided (mapping (day, timestamp) ->
    {product: List[Trade]}), the trader receives populated ``market_trades``
    on every tick (EXACT mode only). Otherwise ``market_trades`` stays empty.
    """
    global FILL_MODE

    # Instantiate trader.
    if trader_cls is DefaultMarketMaker:
        trader = DefaultMarketMaker(params, position_limits)
    else:
        try:
            trader = trader_cls()
            # Set parameters dynamically on the trader instance.
            for k, v in params.items():
                try:
                    setattr(trader, k, v)
                except Exception:
                    pass
            # Some external traders keep parameters in a CONFIG dict.
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
    n_fills = 0
    traderData = ""
    # Per-product running average entry price for round-trip attribution.
    # A round-trip closes when a fill reduces |position| for that product;
    # the realised PnL on the reduced lot (= (exit - avg_entry) * closed_qty,
    # signed by the direction of the old position) is recorded. Position
    # flips through zero are split into a closing lot and a new opening lot.
    avg_entry: Dict[str, float] = defaultdict(float)
    round_trip_pnls: List[float] = []

    def _apply_fill(product: str, qty: int, price: float) -> None:
        """Update avg_entry and append realised PnL to round_trip_pnls.
        ``qty`` uses Prosperity's sign convention: +buy / -sell.
        """
        old_pos = position[product]
        new_pos = old_pos + qty
        # Case 1: opening or adding to an existing position (same sign).
        if old_pos == 0 or (old_pos > 0 and qty > 0) or (old_pos < 0 and qty < 0):
            total_cost = avg_entry[product] * abs(old_pos) + price * abs(qty)
            avg_entry[product] = total_cost / max(abs(new_pos), 1)
            return
        # Case 2: closing or partially closing (opposite sign).
        close_qty = min(abs(qty), abs(old_pos))
        # Sign of realised PnL: long closed by a sell -> (price - avg_entry),
        # short closed by a buy -> (avg_entry - price).
        direction = 1 if old_pos > 0 else -1
        pnl = direction * (price - avg_entry[product]) * close_qty
        round_trip_pnls.append(float(pnl))
        # Case 3: fill flips through zero -> remaining qty opens a new lot.
        remaining = abs(qty) - close_qty
        if remaining > 0:
            avg_entry[product] = price
        elif new_pos == 0:
            avg_entry[product] = 0.0
        # else: partial close, avg_entry on the residual stays the same.

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
            mt_for_tick = (
                market_trades_index.get(k, {}) if market_trades_index else {}
            )
            state = TradingState(
                traderData=traderData,
                timestamp=ts,
                listings={},
                order_depths=depths,
                own_trades={},
                market_trades=mt_for_tick,
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
                _apply_fill(f.product, f.quantity, f.price)
                cash -= f.price * f.quantity  # buy reduces cash
                turnover += abs(f.price * f.quantity)
                position[f.product] += f.quantity
            n_fills += len(fills)
            # Mark-to-market at the current mid.
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
        # --- Pre-compute: forward-fill NaNs, build float64 matrices once. ---
        products = list(mids_by_product.keys())
        clean: Dict[str, np.ndarray] = {}
        for p in products:
            arr = np.asarray(mids_by_product[p], dtype=np.float64)
            if arr.size == 0 or np.all(np.isnan(arr)):
                continue
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

        # Stack into a (P, T) matrix so per-tick MTM becomes a single dot
        # product instead of a Python-level sum over a generator.
        if products and T > 0:
            mid_mat = np.vstack([clean[p][:T] for p in products])  # (P, T)
            # Per-product half-spread vector. ``sim_spread_by_product`` carries
            # the FULL spread in ticks; halving it gives the offset applied to
            # each side of the mid before rounding.
            spread_map = sim_spread_by_product or {}
            half_vec = np.array(
                [
                    max(0.5, float(spread_map.get(p, DEFAULT_SIM_SPREAD)) / 2.0)
                    for p in products
                ],
                dtype=np.float64,
            ).reshape(-1, 1)
            bid_mat = np.floor(mid_mat - half_vec).astype(np.int64)
            ask_mat = np.ceil(mid_mat + half_vec).astype(np.int64)
            # Final safeguard: integer rounding can collapse the book to a
            # crossed/zero spread when half_vec < 1 and mid is exactly on a
            # tick boundary. Force a strict 1-tick separation in that case.
            crossed = ask_mat <= bid_mat
            if crossed.any():
                ask_mat = np.where(crossed, bid_mat + 1, ask_mat)
        else:
            mid_mat = np.zeros((0, 0))
            bid_mat = ask_mat = np.zeros((0, 0), dtype=np.int64)

        # Fast position array kept aligned with ``products``; dict view is
        # rebuilt lazily only when the trader is invoked.
        prod_idx = {p: i for i, p in enumerate(products)}
        pos_arr = np.zeros(len(products), dtype=np.int64)
        # Initialise from any pre-set positions (normally empty).
        for p, v in position.items():
            if p in prod_idx:
                pos_arr[prod_idx[p]] = v

        # Reusable OrderDepth instances -- avoid re-instantiating 1.4M times.
        depth_pool: Dict[str, OrderDepth] = {p: OrderDepth() for p in products}

        for t in range(T):
            # Update the depth pool in place (single level per side).
            depths: Dict[str, OrderDepth] = {}
            for pi, p in enumerate(products):
                m = mid_mat[pi, t]
                if not np.isfinite(m):
                    continue
                od = depth_pool[p]
                od.buy_orders.clear()
                od.sell_orders.clear()
                od.buy_orders[int(bid_mat[pi, t])] = 30
                od.sell_orders[int(ask_mat[pi, t])] = -30
                depths[p] = od

            # Next-mid lookup as ndarray slice (avoid per-product dict comp).
            if t + 1 < T:
                nm_vec = mid_mat[:, t + 1]
                nm = {p: float(nm_vec[pi]) for pi, p in enumerate(products)}
            else:
                nm = {p: None for p in products}

            state = TradingState(
                traderData=traderData,
                timestamp=t * TICK_STEP,
                listings={},
                order_depths=depths,
                own_trades={},
                market_trades={},
                position={p: int(pos_arr[prod_idx[p]]) for p in products},
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
            # Only run full matching when the trader actually submitted orders.
            if orders_out:
                fills, _delta = simulate_fills_one_tick(
                    orders_out, depths,
                    {p: int(pos_arr[prod_idx[p]]) for p in products},
                    position_limits, nm, t * TICK_STEP, "APPROX",
                )
                for f in fills:
                    # BUGFIX: _apply_fill reads position[f.product] to detect
                    # opening vs closing (for round-trip PnL tracking). The
                    # vectorised loop only mutates pos_arr, so without this
                    # sync the position dict stayed 0 forever and every fill
                    # looked like an "opening" -- which made hit_rate_active
                    # and mean_round_trips_per_path collapse to 0 across the
                    # entire grid. We sync BEFORE (so _apply_fill sees the
                    # correct old_pos) and AFTER (so subsequent fills in the
                    # same tick see the updated state).
                    pi = prod_idx.get(f.product)
                    if pi is not None:
                        position[f.product] = int(pos_arr[pi])
                    _apply_fill(f.product, f.quantity, f.price)
                    cash -= f.price * f.quantity
                    turnover += abs(f.price * f.quantity)
                    if pi is not None:
                        pos_arr[pi] += f.quantity
                        position[f.product] = int(pos_arr[pi])
                    else:
                        position[f.product] = position.get(f.product, 0) + f.quantity
                n_fills += len(fills)

            # Vectorised MTM and inventory across products.
            mtm = float(np.dot(pos_arr, mid_mat[:, t]))
            realized_pnl_path.append(cash + mtm)
            inventory_series.append(int(np.abs(pos_arr).sum()))

        # Sync final position dict for any downstream use.
        for pi, p in enumerate(products):
            position[p] = int(pos_arr[pi])

    pnl = np.array(realized_pnl_path) if realized_pnl_path else np.array([0.0])
    final_pnl = float(pnl[-1])
    returns = np.diff(pnl, prepend=0.0)
    max_dd = float((np.maximum.accumulate(pnl) - pnl).max()) if len(pnl) else 0.0
    # Active hit-rate: share of CLOSED round-trips that ended profitable.
    # hit_rate (tick-based, returns>0) is diluted by idle ticks with zero PnL;
    # hit_rate_active ignores them and answers "when I actually finish a
    # round-trip, how often do I win?" -- a much better edge diagnostic.
    rt = np.asarray(round_trip_pnls, dtype=float)
    hit_rate_active = float((rt > 0).mean()) if rt.size else 0.0
    return dict(
        final_pnl=final_pnl,
        mean_step_pnl=float(returns.mean()) if len(returns) else 0.0,
        std_step_pnl=float(returns.std()) if len(returns) > 1 else 0.0,
        max_drawdown=max_dd,
        hit_rate=float((returns > 0).mean()) if len(returns) else 0.0,
        hit_rate_active=hit_rate_active,
        n_round_trips=int(rt.size),
        turnover=float(turnover),
        inventory_std=float(np.std(inventory_series)) if inventory_series else 0.0,
        n_trades=int(n_fills),
        equity_curve=pnl.tolist(),
    )


# =============================================================================
# 10. METRICS OVER MONTE-CARLO PATHS + ROBUST SELECTION
# =============================================================================


def aggregate_path_metrics(
    per_path_profits: np.ndarray,
    risk_lambda: float,
    per_path_trades: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Aggregate metrics across Monte-Carlo paths for a single parameter combo.

    In addition to the standard profit/risk statistics, two
    consistency-oriented metrics are reported:

    * ``sharpe`` -- cross-path Sharpe ratio, ``mean_profit / profit_std``.
      This measures how reliable the PnL is across MC scenarios. It is
      *not* annualised (Prosperity's tick grid has no calendar reference),
      so interpret it as a ratio in profit units, directly comparable
      across combos.
    * ``profit_per_trade`` -- average profit per executed fill
      (``mean_profit / mean_trades_per_path``). A proxy for edge per
      round-trip; higher = more PnL earned per order filled, which
      rewards selective strategies over churn-heavy ones.

    ``objective_score`` stays the risk-adjusted mean
    ``mean - risk_lambda * variance`` to preserve backwards-compatible
    ranking; the new metrics are additive so the fine stage can sort
    on them explicitly.
    """
    if len(per_path_profits) == 0:
        return {}
    mean_p = float(np.mean(per_path_profits))
    med_p = float(np.median(per_path_profits))
    std_p = float(np.std(per_path_profits))
    var_p = float(np.var(per_path_profits))
    var_5 = float(np.quantile(per_path_profits, 0.05))
    cvar_5 = float(per_path_profits[per_path_profits <= var_5].mean()) if np.any(per_path_profits <= var_5) else var_5
    worst = float(per_path_profits.min())

    # Cross-path Sharpe: mean / std. Not annualised (tick grid != time).
    sharpe = mean_p / (std_p + 1e-9)

    # Profit per trade (per executed fill).
    if per_path_trades is not None and len(per_path_trades) > 0:
        mean_trades = float(np.mean(per_path_trades))
        median_trades = float(np.median(per_path_trades))
    else:
        mean_trades = 0.0
        median_trades = 0.0
    profit_per_trade = (mean_p / mean_trades) if mean_trades > 0 else 0.0

    # Normalization (keep the score comparable across grid combinations).
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
        sharpe=sharpe,
        profit_per_trade=profit_per_trade,
        mean_trades_per_path=mean_trades,
        median_trades_per_path=median_trades,
    )


# -----------------------------------------------------------------------------
# Parallel-execution helpers. The per-combo worker must be defined at module
# level so it is picklable by loky / multiprocessing. Each worker process keeps
# a local cache of loaded trader classes to avoid re-importing the trader file
# on every task.
# -----------------------------------------------------------------------------

_WORKER_TRADER_CACHE: Dict[str, Any] = {}
# Per-worker shared state populated by :func:`_worker_init` so large objects
# (MC path arrays, position limits, historical books) are pickled ONCE per
# worker rather than once per job. For a 9504-combo run this turns a pickle
# cost of ~150 GB into ~150 MB -- the single biggest runtime win.
_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(
    trader_path_str: Optional[str],
    mc_paths: Dict[str, np.ndarray],
    hist_mids: Optional[Dict[str, np.ndarray]],
    historical_depths: Optional[Dict],
    ordered_keys: Optional[List],
    position_limits: Dict[str, int],
    max_ticks: Optional[int],
    market_trades_index: Optional[Dict],
    n_paths: int,
    skip_hist_in_grid: bool,
    risk_lambda: float,
    have_exact: bool,
    seed: int,
    lean_metrics: bool = False,
    sim_spread_by_product: Optional[Dict[str, float]] = None,
) -> None:
    """Initializer run ONCE per worker process. Stashes the large per-run
    context into the process-global dict so per-job payloads are tiny.
    """
    global _WORKER_STATE
    _WORKER_STATE = {
        "trader_path_str": trader_path_str,
        "mc_paths": mc_paths,
        "hist_mids": hist_mids,
        "historical_depths": historical_depths,
        "ordered_keys": ordered_keys,
        "sim_spread_by_product": sim_spread_by_product,
        "position_limits": position_limits,
        "max_ticks": max_ticks,
        "market_trades_index": market_trades_index,
        "n_paths": n_paths,
        "skip_hist_in_grid": skip_hist_in_grid,
        "risk_lambda": risk_lambda,
        "have_exact": have_exact,
        "seed": seed,
        "lean_metrics": lean_metrics,
    }
    # Silence noisy NaN warnings once per worker.
    import warnings as _w
    _w.filterwarnings("ignore", category=RuntimeWarning)


def _eval_combo_worker_shared(params: Dict[str, Any], n_eval: int) -> Optional[Dict[str, Any]]:
    """Thin wrapper that reads all heavy inputs from :data:`_WORKER_STATE`.
    Dispatched by :func:`_parallel_map` when workers were initialised with
    shared MC context. Each job only pickles ``(params, n_eval)``.
    """
    st = _WORKER_STATE
    if not st:
        raise RuntimeError("_WORKER_STATE is empty -- initializer did not run")
    return _eval_combo_worker(
        st["trader_path_str"], params, st["mc_paths"], st["hist_mids"],
        st["historical_depths"], st["ordered_keys"], st["position_limits"],
        st["max_ticks"], st["market_trades_index"], n_eval, st["n_paths"],
        st["skip_hist_in_grid"], st["risk_lambda"], st["have_exact"], st["seed"],
        st.get("lean_metrics", False),
        st.get("sim_spread_by_product"),
    )


def _eval_single_path_worker_shared(params: Dict[str, Any], path_idx: int) -> Dict[str, Any]:
    """Shared-state equivalent of :func:`_eval_single_path_worker` for the
    fan-chart and stability stages."""
    st = _WORKER_STATE
    if not st:
        raise RuntimeError("_WORKER_STATE is empty -- initializer was not run")
    return _eval_single_path_worker(
        st["trader_path_str"], params, st["mc_paths"], path_idx,
        st["position_limits"], st["max_ticks"],
        st.get("sim_spread_by_product"),
    )


def _worker_get_trader_cls(trader_path_str: Optional[str]):
    """Return (and cache) the Trader class for a given file path inside a
    worker process. Falls back to :class:`DefaultMarketMaker` on any failure.
    """
    if not trader_path_str:
        return DefaultMarketMaker
    cached = _WORKER_TRADER_CACHE.get(trader_path_str)
    if cached is not None:
        return cached
    cls = load_external_trader(Path(trader_path_str)) or DefaultMarketMaker
    _WORKER_TRADER_CACHE[trader_path_str] = cls
    return cls


def _eval_combo_worker(
    trader_path_str: Optional[str],
    params: Dict[str, Any],
    mc_paths: Dict[str, np.ndarray],
    hist_mids: Optional[Dict[str, np.ndarray]],
    historical_depths: Optional[Dict],
    ordered_keys: Optional[List],
    position_limits: Dict[str, int],
    max_ticks: Optional[int],
    market_trades_index: Optional[Dict],
    n_eval: int,
    n_paths: int,
    skip_hist_in_grid: bool,
    risk_lambda: float,
    have_exact: bool,
    seed: int,
    lean_metrics: bool = False,
    sim_spread_by_product: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate ONE parameter combination across ``n_eval`` Monte-Carlo paths
    (and optionally one historical EXACT backtest). Designed to run inside a
    ``joblib`` worker process -- all inputs must be picklable.

    Returns a flat metrics dict (params merged with aggregated stats) or
    ``None`` if the run raised internally.
    """
    try:
        trader_cls = _worker_get_trader_cls(trader_path_str)

        # Optional historical backtest (disabled by default for grid runs).
        if not skip_hist_in_grid and hist_mids is not None:
            hist_res = run_backtest_on_series(
                trader_cls, params,
                mids_by_product=hist_mids,
                historical_depths=historical_depths if have_exact else None,
                ordered_keys=ordered_keys if have_exact else None,
                position_limits=position_limits,
                max_ticks=max_ticks,
                market_trades_index=market_trades_index if have_exact else None,
                sim_spread_by_product=sim_spread_by_product,
            )
        else:
            hist_res = dict(final_pnl=0.0, max_drawdown=0.0, hit_rate=0.0,
                            hit_rate_active=0.0, n_round_trips=0,
                            turnover=0.0, inventory_std=0.0, equity_curve=[0.0])

        products = list(mc_paths.keys())
        profits: List[float] = []
        trade_counts: List[int] = []
        drawdowns: List[float] = []
        hit_rates: List[float] = []
        hit_rates_active: List[float] = []
        round_trips: List[int] = []
        turnovers: List[float] = []
        inv_stds: List[float] = []
        rng = np.random.default_rng(seed + hash(tuple(sorted(params.items()))) % 2**31)
        n_eval_eff = max(1, min(n_paths, n_eval))
        idx = rng.choice(n_paths, size=n_eval_eff, replace=False)
        for i in idx:
            mids = {p: mc_paths[p][i] for p in products}
            r = run_backtest_on_series(
                trader_cls, params, mids_by_product=mids,
                historical_depths=None, ordered_keys=None,
                position_limits=position_limits,
                max_ticks=max_ticks,
                sim_spread_by_product=sim_spread_by_product,
            )
            profits.append(r["final_pnl"])
            trade_counts.append(r["n_trades"])
            # Per-path risk / activity metrics -- aggregated across MC paths
            # below. Previously these were taken ONLY from hist_res, which is
            # zeroed-out in grid mode (skip_hist_in_grid=True), so every grid
            # row reported 0.0 for drawdown / hit-rate / turnover / inv_std.
            # With --lean-metrics these six list-appends are skipped -- cuts
            # ~5-10% off the per-combo loop for configs that don't use them.
            if not lean_metrics:
                drawdowns.append(float(r.get("max_drawdown", 0.0)))
                hit_rates.append(float(r.get("hit_rate", 0.0)))
                hit_rates_active.append(float(r.get("hit_rate_active", 0.0)))
                round_trips.append(int(r.get("n_round_trips", 0)))
                turnovers.append(float(r.get("turnover", 0.0)))
                inv_stds.append(float(r.get("inventory_std", 0.0)))

        profits_arr = np.array(profits)
        trades_arr = np.array(trade_counts)
        m = aggregate_path_metrics(profits_arr, risk_lambda, per_path_trades=trades_arr)
        # Prefer MC-path aggregates when the historical backtest was skipped
        # (grid mode). Fall back to hist_res only if no MC paths executed.
        def _mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0
        m.update(dict(
            max_drawdown=_mean(drawdowns) if drawdowns else hist_res["max_drawdown"],
            hit_rate=_mean(hit_rates) if hit_rates else hist_res["hit_rate"],
            hit_rate_active=_mean(hit_rates_active) if hit_rates_active else hist_res.get("hit_rate_active", 0.0),
            mean_round_trips_per_path=_mean(round_trips) if round_trips else float(hist_res.get("n_round_trips", 0.0)),
            turnover=_mean(turnovers) if turnovers else hist_res["turnover"],
            inventory_std=_mean(inv_stds) if inv_stds else hist_res["inventory_std"],
            max_drawdown_worst=float(np.max(drawdowns)) if drawdowns else 0.0,
            turnover_per_trade=(_mean(turnovers) / _mean(trade_counts)) if _mean(trade_counts) > 0 else 0.0,
            historical_final_pnl=hist_res["final_pnl"],
            n_eval_paths=n_eval_eff,
        ))
        row = dict(params)
        row.update(m)
        return row
    except Exception as e:
        print(f"[ERR] combo {params} failed in worker: {e}")
        return None


def _eval_single_path_worker(
    trader_path_str: Optional[str],
    params: Dict[str, Any],
    mc_paths: Dict[str, np.ndarray],
    path_idx: int,
    position_limits: Dict[str, int],
    max_ticks: Optional[int],
    sim_spread_by_product: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluate ONE MC path for a fixed parameter combo. Used by the fan-chart
    and stability visualization stages so they also benefit from parallelism.
    """
    trader_cls = _worker_get_trader_cls(trader_path_str)
    products = list(mc_paths.keys())
    mids = {p: mc_paths[p][path_idx] for p in products}
    r = run_backtest_on_series(
        trader_cls, params, mids_by_product=mids,
        historical_depths=None, ordered_keys=None,
        position_limits=position_limits,
        max_ticks=max_ticks,
        sim_spread_by_product=sim_spread_by_product,
    )
    return {"final_pnl": r["final_pnl"], "equity_curve": r["equity_curve"]}


def _parallel_map(
    jobs: List[Any],
    worker_fn: Callable,
    n_workers: int,
    desc: str = "grid",
    show_progress: bool = True,
    initializer: Optional[Callable] = None,
    initargs: Optional[Tuple] = None,
) -> List[Any]:
    """Dispatch a list of delayed jobs across ``n_workers`` processes using
    joblib (``loky`` backend). Falls back to a plain serial loop when joblib
    is not installed or ``n_workers == 1``.

    ``n_workers``:
        * ``-1`` -- use all available CPU cores (default).
        * ``0``  -- treated like ``-1`` for user convenience.
        * ``1``  -- serial execution (legacy behaviour).
        * ``>=2`` -- explicit worker count.
    """
    if n_workers == 0:
        n_workers = -1
    total = len(jobs)
    if total == 0:
        return []
    if n_workers == 1:
        if initializer is not None:
            initializer(*(initargs or ()))
        iterator = tqdm(jobs, desc=desc, total=total) if show_progress else jobs
        return [worker_fn(*args) for args in iterator]
    n_actual = os.cpu_count() or 1 if n_workers < 0 else min(n_workers, len(jobs))

    # When an initializer is provided we route through a reusable process pool
    # directly. joblib's high-level Parallel does not accept initializer, and
    # falling back to per-job pickling would defeat the whole shared-state
    # optimisation. We use ProcessPoolExecutor with a spawn context so worker
    # globals are initialised explicitly via the initializer (loky sometimes
    # respawns workers mid-run and misses late initializer dispatches for the
    # replacement, which we have seen in practice).
    if initializer is not None:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Cache the executor on the initializer identity + args tuple id so
        # repeated _parallel_map calls within one pipeline reuse workers
        # (mc_paths is pickled ONCE per worker for the whole run).
        cache_key = (id(initializer), id(initargs))
        pool_cache = getattr(_parallel_map, "_pool_cache", None)
        if pool_cache is None:
            pool_cache = {}
            _parallel_map._pool_cache = pool_cache  # type: ignore[attr-defined]
        executor = pool_cache.get(cache_key)
        if executor is None:
            ctx = _mp.get_context("spawn")
            executor = ProcessPoolExecutor(
                max_workers=n_actual,
                initializer=initializer,
                initargs=initargs or (),
                mp_context=ctx,
            )
            pool_cache[cache_key] = executor
        print(f"[PAR] {desc}: dispatching {total} jobs to {n_actual} workers (spawn, shared state)")
        results: List[Any] = [None] * total
        futures = {executor.submit(worker_fn, *args): i for i, args in enumerate(jobs)}
        if show_progress and _HAS_TQDM:
            with tqdm(total=total, desc=desc) as pbar:
                for f in as_completed(futures):
                    idx = futures[f]
                    results[idx] = f.result()
                    pbar.update(1)
        else:
            for f in as_completed(futures):
                idx = futures[f]
                results[idx] = f.result()
        return results

    if not _HAS_JOBLIB:
        # Serial fallback with initializer behaviour.
        if initializer is not None:
            initializer(*(initargs or ()))
        iterator = tqdm(jobs, desc=desc, total=total) if show_progress else jobs
        return [worker_fn(*args) for args in iterator]

    print(f"[PAR] {desc}: dispatching {total} jobs to {n_actual} workers (joblib/loky)")
    parallel_kwargs = dict(n_jobs=n_workers, backend="loky")
    if show_progress and _HAS_TQDM:
        with tqdm(total=total, desc=desc) as pbar:
            results = [None] * total

            def _wrapped(idx, args):
                r = worker_fn(*args)
                return idx, r

            try:
                gen = Parallel(return_as="generator", **parallel_kwargs)(
                    delayed(_wrapped)(i, args) for i, args in enumerate(jobs)
                )
                for idx, r in gen:
                    results[idx] = r
                    pbar.update(1)
                return results
            except TypeError:
                out = Parallel(**parallel_kwargs)(
                    delayed(worker_fn)(*args) for args in jobs
                )
                pbar.update(total)
                return list(out)
    else:
        return list(
            Parallel(**parallel_kwargs)(
                delayed(worker_fn)(*args) for args in jobs
            )
        )


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, maximize_y: bool = True) -> pd.DataFrame:
    """Pareto frontier: we want mean_profit MAX and variance MIN.
    Here x=variance (min), y=mean_profit (max)."""
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


def pick_top2_robust(full_results: pd.DataFrame, metric: str = "sharpe") -> pd.DataFrame:
    """Robust Top-2 selection combining global score and neighbourhood
    stability. Stability is measured as the mean ranking metric across the
    nearest neighbours in (normalised) parameter space.
    """
    if full_results.empty:
        return full_results
    df = full_results.copy()
    # Fall back cleanly if the requested metric column is missing.
    if metric not in df.columns:
        metric = "objective_score" if "objective_score" in df.columns else df.columns[0]
    # Filter out non-finite metric values so std/mean do not blow up.
    df = df[np.isfinite(df[metric])].reset_index(drop=True)
    if df.empty:
        return df
    # Normalise for the robust score.
    df["obj_norm"] = (df[metric] - df[metric].mean()) / (df[metric].std() + 1e-9)
    # Neighbour score: mean objective of the K nearest neighbours in
    # numeric-parameter space (Euclidean on z-scored columns).
    num_cols = [
        c for c in df.columns
        if c not in {
            "objective_score", "obj_norm", "total_profit", "mean_profit", "median_profit",
            "profit_std", "variance", "VaR_5", "CVaR_5", "worst_path_profit", "max_drawdown",
            "max_drawdown_worst", "turnover_per_trade",
            "hit_rate", "hit_rate_active", "mean_round_trips_per_path",
            "turnover", "inventory_std",
            "sharpe", "profit_per_trade", "mean_trades_per_path", "median_trades_per_path",
            "n_trades", "historical_final_pnl", "n_eval_paths",
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


def plot_top_ranking(df: pd.DataFrame, out: Path, n: int = 20, metric: str = "sharpe"):
    if df.empty:
        return
    if metric not in df.columns:
        metric = "objective_score" if "objective_score" in df.columns else df.columns[0]
    top = df.nlargest(n, metric).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top)), top[metric])
    ax.set_title(f"Top-{n} Parameter Combinations by {metric}")
    ax.set_xlabel("rank"); ax.set_ylabel(metric)
    save_plot(fig, out)


def plot_profit_hit_map(df: pd.DataFrame, spec: Dict[str, Any], out: Path,
                        metric: str = "sharpe"):
    """2D heat-map over the two highest-cardinality numeric parameters, cells
    coloured by the mean of ``metric`` (averaged across the other dims).
    Skips silently if fewer than 2 numeric params are available.
    """
    if df.empty:
        return
    if metric not in df.columns:
        metric = "objective_score" if "objective_score" in df.columns else None
    if metric is None:
        return
    numeric_params = [
        k for k, v in spec.items()
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k])
    ]
    if len(numeric_params) < 2:
        return
    ranked = sorted(numeric_params, key=lambda k: -df[k].nunique())
    k1, k2 = ranked[0], ranked[1]
    piv = df.pivot_table(index=k1, columns=k2, values=metric, aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:g}" if isinstance(c, (int, float)) else str(c)
                        for c in piv.columns], rotation=45)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{i:g}" if isinstance(i, (int, float)) else str(i)
                        for i in piv.index])
    ax.set_xlabel(k2); ax.set_ylabel(k1)
    ax.set_title(f"Profit hit-map: mean {metric} by {k1} x {k2}")
    fig.colorbar(im, ax=ax, label=f"mean {metric}")
    save_plot(fig, out)


def plot_parameter_sensitivity(df: pd.DataFrame, spec: Dict[str, Any], out: Path,
                               metric: str = "sharpe"):
    """Grid of 1D sensitivity plots: for each numeric parameter in ``spec``,
    plot mean(metric) +/- std across all grid rows at each value of that
    parameter. Useful to spot monotone vs peaked responses.
    """
    if df.empty:
        return
    if metric not in df.columns:
        metric = "objective_score" if "objective_score" in df.columns else None
    if metric is None:
        return
    numeric_params = [
        k for k, v in spec.items()
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k])
    ]
    if not numeric_params:
        return
    n = len(numeric_params)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows),
                             squeeze=False)
    for i, key in enumerate(numeric_params):
        ax = axes[i // cols][i % cols]
        grp = df.groupby(key)[metric].agg(["mean", "std", "count"]).reset_index()
        grp = grp.sort_values(key)
        x = grp[key].values
        mu = grp["mean"].values
        sd = grp["std"].fillna(0).values
        ax.plot(x, mu, marker="o", lw=1.4)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
        ax.set_title(f"{key}")
        ax.set_xlabel(key); ax.set_ylabel(f"mean {metric}")
        ax.grid(alpha=0.3)
    # Blank out unused subplots.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Parameter sensitivity: mean {metric} vs parameter value",
                 y=1.02, fontsize=12)
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


# =============================================================================
#  BLACK-SCHOLES + VOLATILITY SMILE (Prosperity-4 Round 3 -- VEV options)
# =============================================================================
#
#  Two distinct concepts that MUST NOT be mixed:
#    * TTE_days        -- integer Solvenarian days until expiry (8, 7, 6, 5...).
#                         This is *time of expiration* -- a calendar countdown.
#    * T (T_years)     -- time to MATURITY, in years, the BS formula's input.
#                         T_years = TTE_days_remaining / DAYS_PER_YEAR.
#
#  For our P4-R3 dataset (see problem statement):
#    historical day 0 (tutorial round)  -> TTE_days starts at 8
#    historical day 1 (Round 1)         -> TTE_days starts at 7
#    historical day 2 (Round 2)         -> TTE_days starts at 6
#    live  Round 3 submission           -> TTE_days starts at 5
#
#  So the per-tick remaining TTE in Solvenarian days is
#      tte_days_remaining(day, timestamp) = TTE_BASE_DAYS - day
#                                           - timestamp / TICKS_PER_DAY
#  where TICKS_PER_DAY = 1_000_000 (Prosperity timestamps step by 100 up to
#  999_900 per day). We divide by DAYS_PER_YEAR=365 to get T_years for BS.
#
#  Black-Scholes assumptions for this competition (user-confirmed):
#    r (risk-free rate)    = 0.0  (annualised, continuously compounded)
#    q (dividend yield)    = 0.0
#    option style          = European call
#    no early exercise
# =============================================================================

DAYS_PER_YEAR: int = 365               # annualisation factor for BS T_years
TICKS_PER_DAY: int = 1_000_000         # max Prosperity timestamp per day + 100
# Default TTE at the start of historical day 0 for the Round 3 dataset. The
# user can override this via --tte-base-days when the dataset corresponds to a
# different round (e.g. for a Round 4 dataset where historical day 0 starts at
# TTE=6 because two live rounds have already been consumed).
DEFAULT_TTE_BASE_DAYS: int = 8


def _std_norm_cdf(x: float) -> float:
    if _HAS_SCIPY:
        return float(_scipy_norm.cdf(x))
    # Abramowitz-Stegun 7.1.26 approximation; max error ~1.5e-7. Sufficient
    # for IV solving, though scipy is preferred when available.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _std_norm_pdf(x: float) -> float:
    if _HAS_SCIPY:
        return float(_scipy_norm.pdf(x))
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European call price. r=0, no dividends.

    Parameters
    ----------
    S : underlying spot price
    K : strike price
    T : time to MATURITY in years (NOT TTE in Solvenarian days)
    sigma : annualised implied volatility (same time unit as T)
    r : risk-free rate, continuously compounded (always 0 here)
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(0.0, S - K * math.exp(-r * max(T, 0.0)))
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _std_norm_cdf(d1) - K * math.exp(-r * T) * _std_norm_cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes vega dC/dsigma. Used by Newton's method for IV."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return S * _std_norm_pdf(d1) * sqrt_T


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Delta of a European call (N(d1))."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return _std_norm_cdf(d1)


def bs_implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 5.0,
) -> Optional[float]:
    """Invert BS for sigma: Newton-Raphson with bisection fallback.

    Returns None when:
      * market price is below intrinsic value (no real IV),
      * T <= 0 (option already expired),
      * the solver does not converge inside [sigma_lo, sigma_hi].

    r=0, no dividends.
    """
    if T <= 0.0 or S <= 0.0 or K <= 0.0 or market_price <= 0.0:
        return None
    intrinsic = max(0.0, S - K * math.exp(-r * T))
    # Require *strictly positive* extrinsic so vega is meaningful.
    if market_price <= intrinsic + 1e-10:
        return None
    # Brenner-Subrahmanyam seed (close-to-ATM) gives a robust starting point.
    sigma = max(sigma_lo, min(sigma_hi,
                              math.sqrt(2.0 * math.pi / T) * market_price / S))
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, sigma, r)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        vega = bs_vega(S, K, T, sigma, r)
        if vega < 1e-10:
            break  # fall through to bisection
        sigma_next = sigma - diff / vega
        if not math.isfinite(sigma_next):
            break
        sigma = max(sigma_lo, min(sigma_hi, sigma_next))
    # Bisection fallback on [sigma_lo, sigma_hi].
    lo, hi = sigma_lo, sigma_hi
    f_lo = bs_call_price(S, K, T, lo, r) - market_price
    f_hi = bs_call_price(S, K, T, hi, r) - market_price
    if f_lo * f_hi > 0:
        return None  # no root in the interval
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_price(S, K, T, mid, r) - market_price
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def tte_days_remaining(
    day: int, timestamp: int, tte_base_days: int = DEFAULT_TTE_BASE_DAYS
) -> float:
    """Time TO EXPIRATION in *Solvenarian days* (not years).

    NOT a valid BS input on its own -- convert to years via
    :func:`tte_to_maturity_years`.
    """
    return max(0.0, tte_base_days - day - timestamp / TICKS_PER_DAY)


def tte_to_maturity_years(
    day: int,
    timestamp: int,
    tte_base_days: int = DEFAULT_TTE_BASE_DAYS,
    days_per_year: int = DAYS_PER_YEAR,
) -> float:
    """Time to MATURITY in years (valid BS input)."""
    return tte_days_remaining(day, timestamp, tte_base_days) / days_per_year


_VEV_STRIKE_RE = re.compile(r"^(?:VEV|VELVETFRUIT_EXTRACT_VOUCHER)_(\d+)$")


def extract_voucher_strike(product: str) -> Optional[float]:
    """Return the integer strike embedded in a VEV / full voucher product name.

    Handles both the short alias ``VEV_5000`` and the official full name
    ``VELVETFRUIT_EXTRACT_VOUCHER_5000``. Returns None for non-option products.
    """
    m = _VEV_STRIKE_RE.match(product)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _resolve_underlying_column(products: Iterable[str]) -> Optional[str]:
    """Return the underlying product name if present in the dataset."""
    candidates = ("VELVETFRUIT_EXTRACT", "VOLCANIC_ROCK")  # P4-R3 / P3-R4 fallback
    prods = set(products)
    for c in candidates:
        if c in prods:
            return c
    return None


def build_vol_smile_table(
    prices_df: pd.DataFrame,
    tte_base_days: int = DEFAULT_TTE_BASE_DAYS,
    days_per_year: int = DAYS_PER_YEAR,
    min_extrinsic: float = 0.0,
    min_T: float = 1e-6,
) -> pd.DataFrame:
    """Per-tick IV + moneyness table for every voucher in prices_df.

    Vectorised end-to-end (no per-row Python loop over the joined frame).
    Drops ticks with non-positive extrinsic value -- those are the
    bottom-left outliers from figure 6a. Columns of the returned frame:
        day, timestamp, product, K, S, market_price,
        TTE_days, T_years, m_t, IV
    """
    products = list(prices_df["product"].unique())
    underlying = _resolve_underlying_column(products)
    if underlying is None:
        print("[VOL-SMILE] No recognisable underlying found (expected "
              "VELVETFRUIT_EXTRACT); skipping.")
        return pd.DataFrame()
    vouchers = [(p, extract_voucher_strike(p)) for p in products]
    vouchers = [(p, k) for p, k in vouchers if k is not None]
    if not vouchers:
        print("[VOL-SMILE] No voucher products (VEV_*) found; skipping.")
        return pd.DataFrame()

    und = (prices_df[prices_df["product"] == underlying]
           [["day", "timestamp", "mid_price"]]
           .rename(columns={"mid_price": "S"}))
    rows: List[pd.DataFrame] = []
    for prod, strike in vouchers:
        sub = prices_df[prices_df["product"] == prod][[
            "day", "timestamp", "mid_price"]].rename(columns={"mid_price": "market_price"})
        if sub.empty:
            continue
        merged = sub.merge(und, on=["day", "timestamp"], how="inner")
        merged = merged.dropna(subset=["S", "market_price"])
        merged = merged[(merged["S"] > 0) & (merged["market_price"] > 0)]
        if merged.empty:
            continue
        merged["product"] = prod
        merged["K"] = float(strike)
        # TTE vectorised.
        tte_days = (tte_base_days
                    - merged["day"].to_numpy()
                    - merged["timestamp"].to_numpy() / TICKS_PER_DAY)
        T_years = np.maximum(tte_days, 0.0) / days_per_year
        merged["TTE_days"] = tte_days
        merged["T_years"] = T_years
        # Keep ticks with non-negative extrinsic AND T > 0. Note: deep-OTM
        # options often sit at the 0.5 tick floor (extrinsic == 0.5 exactly),
        # so the comparison is >= rather than > to keep them.
        intrinsic = np.maximum(merged["S"].to_numpy() - merged["K"].to_numpy(), 0.0)
        extrinsic = merged["market_price"].to_numpy() - intrinsic
        mask = (T_years > min_T) & (extrinsic >= min_extrinsic)
        merged = merged.loc[mask].reset_index(drop=True)
        if merged.empty:
            continue
        # Moneyness m_t = log(K / S) / sqrt(T_years). Uses T (years), NOT TTE_days.
        merged["m_t"] = (np.log(merged["K"].to_numpy() / merged["S"].to_numpy())
                         / np.sqrt(merged["T_years"].to_numpy()))
        # IV solved per row (no vector BS inverse; fast enough for ~60k rows).
        ivs = np.empty(len(merged), dtype=float)
        S_arr = merged["S"].to_numpy()
        K_arr = merged["K"].to_numpy()
        P_arr = merged["market_price"].to_numpy()
        T_arr = merged["T_years"].to_numpy()
        for i in range(len(merged)):
            iv = bs_implied_vol(P_arr[i], S_arr[i], K_arr[i], T_arr[i], r=0.0)
            ivs[i] = iv if (iv is not None and iv > 0) else np.nan
        merged["IV"] = ivs
        merged = merged.dropna(subset=["IV"])
        if not merged.empty:
            rows.append(merged)

    if not rows:
        print("[VOL-SMILE] IV extraction produced no valid rows; skipping.")
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[["day", "timestamp", "product", "K", "S",
               "market_price", "TTE_days", "T_years", "m_t", "IV"]]
    return out


def fit_vol_smile_parabola(
    smile_table: pd.DataFrame,
    fit_m_min: float = -1.8,
    fit_m_max: float = 1.8,
) -> Tuple[Optional[np.ndarray], pd.DataFrame]:
    """Fit IV ~ a*m^2 + b*m + c on the IV/moneyness table.

    Deep ITM/OTM ticks (|m_t| outside [fit_m_min, fit_m_max]) are excluded
    from the regression because their IVs are numerically unreliable
    (low vega for deep ITM, tick-floor pricing for deep OTM). They are
    still kept in the returned table so the scatter plot shows them.

    Augments ``smile_table`` with:
        smile_IV          : fitted v_hat at each tick's m_t
        iv_deviation      : IV - smile_IV      (figure 6b signal)
        bs_theo_price     : BS(S, K, T, smile_IV, r=0)
        price_deviation   : market_price - bs_theo_price   (figure 6c signal)
        in_fit_window     : bool, True if the row was used in the regression
    Returns (coeffs, augmented_table). coeffs=None if fit failed.
    """
    if smile_table.empty:
        return None, smile_table
    m = smile_table["m_t"].to_numpy()
    iv = smile_table["IV"].to_numpy()
    finite = np.isfinite(m) & np.isfinite(iv)
    in_window = finite & (m >= fit_m_min) & (m <= fit_m_max)
    if in_window.sum() < 10:
        print(f"[VOL-SMILE] Too few valid IV points in fit window "
              f"[{fit_m_min}, {fit_m_max}] (got {int(in_window.sum())}); "
              f"falling back to all finite points.")
        in_window = finite
        if in_window.sum() < 10:
            print("[VOL-SMILE] Too few valid IV points for a degree-2 fit.")
            return None, smile_table
    try:
        coeffs = np.polyfit(m[in_window], iv[in_window], deg=2)
    except Exception as exc:
        print(f"[VOL-SMILE] polyfit failed: {exc}")
        return None, smile_table
    n_total = int(finite.sum())
    n_used = int(in_window.sum())
    n_excl = n_total - n_used
    if n_excl > 0:
        print(f"[VOL-SMILE] parabola fit: {n_used}/{n_total} points used "
              f"({n_excl} deep ITM/OTM excluded, |m_t| outside "
              f"[{fit_m_min}, {fit_m_max}])")
    poly = np.poly1d(coeffs)
    smile_table = smile_table.copy()
    smile_table["smile_IV"] = poly(smile_table["m_t"].to_numpy())
    smile_table["iv_deviation"] = smile_table["IV"] - smile_table["smile_IV"]
    theo = np.empty(len(smile_table), dtype=float)
    S_arr = smile_table["S"].to_numpy()
    K_arr = smile_table["K"].to_numpy()
    T_arr = smile_table["T_years"].to_numpy()
    IV_arr = smile_table["smile_IV"].to_numpy()
    for i in range(len(smile_table)):
        s = IV_arr[i] if (np.isfinite(IV_arr[i]) and IV_arr[i] > 0) else 1e-4
        theo[i] = bs_call_price(S_arr[i], K_arr[i], T_arr[i], s, r=0.0)
    smile_table["bs_theo_price"] = theo
    smile_table["price_deviation"] = smile_table["market_price"] - smile_table["bs_theo_price"]
    smile_table["in_fit_window"] = (
        np.isfinite(smile_table["m_t"].to_numpy())
        & np.isfinite(smile_table["IV"].to_numpy())
        & (smile_table["m_t"].to_numpy() >= fit_m_min)
        & (smile_table["m_t"].to_numpy() <= fit_m_max)
    )
    return coeffs, smile_table


def plot_vol_smile_scatter(
    table: pd.DataFrame, coeffs: np.ndarray, out: Path
) -> None:
    """Figure 6a: IV vs moneyness scatter, colored by strike, fitted parabola."""
    if table.empty or coeffs is None:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    strikes = sorted(table["K"].unique())
    cmap = plt.get_cmap("tab20")
    has_window_col = "in_fit_window" in table.columns
    for i, k in enumerate(strikes):
        sub = table[table["K"] == k]
        if has_window_col:
            in_w = sub[sub["in_fit_window"]]
            out_w = sub[~sub["in_fit_window"]]
            ax.scatter(in_w["m_t"], in_w["IV"], s=4, alpha=0.6,
                       color=cmap(i % 20), label=f"strike={int(k)}")
            if len(out_w) > 0:
                ax.scatter(out_w["m_t"], out_w["IV"], s=4, alpha=0.25,
                           color=cmap(i % 20), marker="x")
        else:
            ax.scatter(sub["m_t"], sub["IV"], s=4, alpha=0.5,
                       color=cmap(i % 20), label=f"strike={int(k)}")
    # Parabola overlay on the observed moneyness range.
    m_min, m_max = table["m_t"].min(), table["m_t"].max()
    m_line = np.linspace(m_min, m_max, 400)
    ax.plot(m_line, np.poly1d(coeffs)(m_line),
            color="black", lw=2.5, label="fitted parabola")
    ax.set_xlabel("m_t = log(K/S) / sqrt(T_years)")
    ax.set_ylabel("v_t (implied vol, annualized)")
    ax.set_title(f"VEV volatility smile  "
                 f"(fit: {coeffs[0]:.4f} m^2 + {coeffs[1]:+.4f} m + {coeffs[2]:+.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=3, loc="best", fontsize=8)
    save_plot(fig, out)


def plot_iv_deviation_ts(table: pd.DataFrame, out: Path) -> None:
    """Figure 6b: time series of IV - smile_IV, one line per strike."""
    if table.empty or "iv_deviation" not in table.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    # Build a continuous timestamp that spans all days for readability.
    t_cont = table["day"].to_numpy() * TICKS_PER_DAY + table["timestamp"].to_numpy()
    tmp = table.copy()
    tmp["t_cont"] = t_cont
    cmap = plt.get_cmap("tab10")
    for i, k in enumerate(sorted(tmp["K"].unique())):
        sub = tmp[tmp["K"] == k].sort_values("t_cont")
        ax.plot(sub["t_cont"], sub["iv_deviation"], lw=0.6, alpha=0.8,
                color=cmap(i % 10), label=f"strike={int(k)}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("timestamp (day-continuous)")
    ax.set_ylabel("Option_IV - VolSmile_IV")
    ax.set_title("IV deviation vs fitted smile (figure 6b)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    save_plot(fig, out)


def plot_price_deviation_ts(table: pd.DataFrame, out: Path) -> None:
    """Figure 6c: time series of market_price - bs_theo_price, per strike."""
    if table.empty or "price_deviation" not in table.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    t_cont = table["day"].to_numpy() * TICKS_PER_DAY + table["timestamp"].to_numpy()
    tmp = table.copy()
    tmp["t_cont"] = t_cont
    cmap = plt.get_cmap("tab10")
    for i, k in enumerate(sorted(tmp["K"].unique())):
        sub = tmp[tmp["K"] == k].sort_values("t_cont")
        ax.plot(sub["t_cont"], sub["price_deviation"], lw=0.6, alpha=0.8,
                color=cmap(i % 10), label=f"strike={int(k)}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("timestamp (day-continuous)")
    ax.set_ylabel("Option_Price - BS_theo(VolSmile_IV)")
    ax.set_title("Voucher price deviation vs BS theoretical (figure 6c)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    save_plot(fig, out)


def run_vol_smile_analysis(
    prices_df: pd.DataFrame,
    out_dir: Path,
    tte_base_days: int = DEFAULT_TTE_BASE_DAYS,
    days_per_year: int = DAYS_PER_YEAR,
    min_extrinsic: float = 0.0,
    strike_groups: Optional[List[Tuple[str, List[int]]]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """End-to-end: IV table -> parabola fit(s) -> CSV + plots.

    If ``strike_groups`` is provided as a list of (label, [strikes]) pairs, an
    independent parabola is fit per group (e.g. ATM core vs. wings), each
    with its own scatter PNG and JSON entry. The 'all' fit on the full table
    is always produced as a baseline.

    Returns a dict {group_label -> coeffs}, or None if no IVs could be
    computed at all.
    """
    print(f"[VOL-SMILE] computing IV + moneyness table "
          f"(tte_base_days={tte_base_days}, days_per_year={days_per_year}, "
          f"min_extrinsic={min_extrinsic})")
    table = build_vol_smile_table(
        prices_df, tte_base_days=tte_base_days,
        days_per_year=days_per_year, min_extrinsic=min_extrinsic,
    )
    if table.empty:
        return None
    coeffs_all, table = fit_vol_smile_parabola(table)
    if coeffs_all is None:
        return None
    print(f"[VOL-SMILE] [all] parabola coeffs (a, b, c) = "
          f"[{coeffs_all[0]:.6f}, {coeffs_all[1]:.6f}, {coeffs_all[2]:.6f}]  "
          f"(fit on {int(table['in_fit_window'].sum())} ticks)")

    # -- Baseline 'all' artefacts (back-compat).
    table.to_csv(out_dir / "vol_smile_per_tick.csv", index=False)
    plot_vol_smile_scatter(table, coeffs_all, out_dir / "plot_vol_smile_scatter.png")
    plot_iv_deviation_ts(table, out_dir / "plot_iv_deviation_ts.png")
    plot_price_deviation_ts(table, out_dir / "plot_price_deviation_ts.png")

    fits: Dict[str, np.ndarray] = {"all": np.asarray(coeffs_all)}
    fit_meta: Dict[str, Dict[str, Any]] = {
        "all": {
            "strikes": sorted(int(k) for k in table["K"].unique()),
            "n_ticks": int(len(table)),
            "n_fit": int(table["in_fit_window"].sum()),
            "coeffs_highest_first": [float(x) for x in coeffs_all],
            "m_t_range": [float(table["m_t"].min()), float(table["m_t"].max())],
            "IV_range": [float(table["IV"].min()), float(table["IV"].max())],
        }
    }

    # -- Per-group parabolas.
    if strike_groups:
        for label, strikes in strike_groups:
            label_safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_") or "group"
            strikes_set = {int(k) for k in strikes}
            sub = table[table["K"].isin(strikes_set)].copy()
            if sub.empty:
                print(f"[VOL-SMILE] [{label}] no rows for strikes {sorted(strikes_set)}; skipping group.")
                continue
            # Drop the prior 'all' fit columns so the per-group fit re-augments cleanly.
            sub = sub.drop(columns=["smile_IV", "iv_deviation", "bs_theo_price",
                                    "price_deviation", "in_fit_window"], errors="ignore")
            coeffs_g, sub_aug = fit_vol_smile_parabola(sub)
            if coeffs_g is None:
                print(f"[VOL-SMILE] [{label}] fit failed; skipping group.")
                continue
            n_fit_g = int(sub_aug["in_fit_window"].sum())
            print(f"[VOL-SMILE] [{label}] strikes={sorted(strikes_set)}  "
                  f"coeffs (a, b, c) = [{coeffs_g[0]:.6f}, {coeffs_g[1]:.6f}, {coeffs_g[2]:.6f}]  "
                  f"(fit on {n_fit_g} ticks)")
            sub_aug.to_csv(out_dir / f"vol_smile_per_tick_{label_safe}.csv", index=False)
            plot_vol_smile_scatter(
                sub_aug, coeffs_g, out_dir / f"plot_vol_smile_scatter_{label_safe}.png"
            )
            fits[label_safe] = np.asarray(coeffs_g)
            fit_meta[label_safe] = {
                "strikes": sorted(int(k) for k in sub_aug["K"].unique()),
                "n_ticks": int(len(sub_aug)),
                "n_fit": n_fit_g,
                "coeffs_highest_first": [float(x) for x in coeffs_g],
                "m_t_range": [float(sub_aug["m_t"].min()), float(sub_aug["m_t"].max())],
                "IV_range": [float(sub_aug["IV"].min()), float(sub_aug["IV"].max())],
            }

    with open(out_dir / "vol_smile_fit.json", "w") as f:
        json.dump({
            "tte_base_days": tte_base_days,
            "days_per_year": days_per_year,
            "min_extrinsic": min_extrinsic,
            "r": 0.0,
            "dividend_yield": 0.0,
            # Back-compat: top-level fields mirror the 'all' fit.
            "parabola_coeffs_highest_first": [float(x) for x in coeffs_all],
            "n_ticks": int(len(table)),
            "m_t_range": [float(table["m_t"].min()), float(table["m_t"].max())],
            "IV_range": [float(table["IV"].min()), float(table["IV"].max())],
            "groups": fit_meta,
        }, f, indent=2)
    return fits


def _single_day_keys(
    ordered_keys: List[Tuple[int, int]], exact_days: int
) -> List[Tuple[int, int]]:
    """Restrict EXACT mode to the first ``exact_days`` days (deterministic)."""
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
    n_workers: int = -1,
    rank_metric: str = "sharpe",
    min_trades_filter: float = 5.0,
    mc_method: str = "bootstrap",
    ou_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    mc_method_overrides: Optional[Dict[str, str]] = None,
    position_limit_overrides: Optional[Dict[str, int]] = None,
    skip_stability: bool = False,
    skip_extra_plots: bool = False,
    lean_metrics: bool = False,
    skip_final_reval: bool = False,
    param_grid_overrides: Optional[Dict[str, List[Any]]] = None,
    tte_base_days: int = DEFAULT_TTE_BASE_DAYS,
    skip_vol_smile: bool = False,
    sim_spread_overrides: Optional[Dict[str, float]] = None,
    sim_spread_mode: str = "auto",
    vol_smile_groups: Optional[List[Tuple[str, List[int]]]] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # -- Parallel-execution banner.
    if n_workers != 1:
        eff = (os.cpu_count() or 1) if n_workers < 1 else n_workers
        print(f"[INFO] parallel mode: spawn + shared-state, n_workers={n_workers} "
              f"(effective ~{eff})")
    else:
        print("[INFO] parallel mode: off (single-threaded)")

    # -- Discover & load.
    files = discover_files(data_dir, round_filter=round_filter)
    print(f"[INFO] prices={len(files['prices'])} trades={len(files['trades'])} others={len(files['others'])}")

    prices_df = load_prices(files["prices"])
    trades_df = load_trades(files["trades"])

    # Fallback: if no round-specific match, accept generic CSVs with at
    # least the columns {timestamp, product, mid_price}.
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
        # ASSUMPTION: no data found. Generate a synthetic 3-day, 1-product
        # dataset so the pipeline still runs end-to-end and the Monte-Carlo
        # section remains testable.
        print("")
        print("=" * 78)
        print("[ERROR] No Prosperity price CSVs were found under --data-dir.")
        print(f"        data_dir resolved to: {data_dir.resolve() if data_dir.exists() else data_dir}")
        print(f"        exists={data_dir.exists()}  is_dir={data_dir.is_dir() if data_dir.exists() else False}")
        print("        Expected files like  prices_round_<R>_day_<D>.csv  /  trades_round_<R>_day_<D>.csv")
        print("        Falling back to a SYNTHETIC 1-product dataset ('SYNTH_PRODUCT').")
        print("        Most external traders won't produce trades on SYNTH_PRODUCT, so grid CSVs")
        print("        will be empty after --min-trades-filter. Re-run with a correct --data-dir.")
        print("=" * 78)
        print("")
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

    # -- Volatility-smile analysis (VEV vouchers, Black-Scholes with r=0,
    # no dividends). Auto-skipped when no VEV products are in the dataset.
    if not skip_vol_smile:
        try:
            vev_mask = prices_df["product"].astype(str).str.match(
                r"^(?:VEV|VELVETFRUIT_EXTRACT_VOUCHER)_\d+$"
            )
            if vev_mask.any():
                print(f"[VOL-SMILE] detected {int(vev_mask.sum())} voucher rows -- running analysis")
                run_vol_smile_analysis(
                    prices_df, out_dir=out_dir, tte_base_days=tte_base_days,
                    strike_groups=vol_smile_groups,
                )
            else:
                print("[VOL-SMILE] no VEV_* / VELVETFRUIT_EXTRACT_VOUCHER_* products -- skipping")
        except Exception as exc:  # pragma: no cover
            # Diagnostic only -- must never crash the main pipeline.
            print(f"[VOL-SMILE] analysis failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    else:
        print("[VOL-SMILE] analysis skipped (--skip-vol-smile)")

    # -- Features.
    features = build_feature_frame(prices_df)
    diag = feature_diagnostics(features)
    diag.to_csv(out_dir / "feature_diagnostics.csv", index=False)

    # -- Position limits.
    position_limits = infer_position_limits(prices_df, overrides=position_limit_overrides)
    print(f"[INFO] inferred position limits: {position_limits}")

    # -- Simulated bid-ask spread (APPROX-fill mode).
    # Three modes:
    #   "auto"    -> per-product mean of historical (ask_price_1 - bid_price_1),
    #                fallback DEFAULT_SIM_SPREAD when L1 quotes are missing.
    #   "legacy"  -> uniform DEFAULT_SIM_SPREAD = 2.0 for every product
    #                (reproduces the original floor(mid-1)/ceil(mid+1) book).
    #   "global"  -> uniform value supplied via --sim-spread N (single number).
    # Per-product overrides (--sim-spread PRODUCT=N) always take precedence.
    sim_spread_mode_norm = (sim_spread_mode or "auto").lower()
    if sim_spread_mode_norm == "legacy":
        sim_spread_by_product = {
            str(p): float(DEFAULT_SIM_SPREAD)
            for p in prices_df["product"].unique()
        }
    else:
        sim_spread_by_product = compute_mean_historical_spread(prices_df)
        # Backfill products that weren't in the historical frame yet (rare,
        # e.g. when MC paths get generated for products lacking L1 quotes).
        for p in position_limits.keys():
            sim_spread_by_product.setdefault(str(p), float(DEFAULT_SIM_SPREAD))
    if sim_spread_overrides:
        # A bare numeric value on --sim-spread is encoded under the special
        # '__GLOBAL__' key by _parse_sim_spread; expand it across every
        # known product BEFORE per-product overrides land so the latter win.
        ovr = dict(sim_spread_overrides)
        global_val = ovr.pop("__GLOBAL__", None)
        if global_val is not None:
            gv = float(global_val)
            for p in list(sim_spread_by_product.keys()):
                sim_spread_by_product[p] = gv
            for p in position_limits.keys():
                sim_spread_by_product[str(p)] = gv
            sim_spread_mode_norm = "global"
        for k, v in ovr.items():
            sim_spread_by_product[str(k)] = float(v)
    if sim_spread_by_product:
        # Compact log: show first ~6 products to keep output readable.
        preview = ", ".join(
            f"{p}={sim_spread_by_product[p]:.2f}"
            for p in list(sim_spread_by_product.keys())[:6]
        )
        more = f" (+{len(sim_spread_by_product)-6} more)" if len(sim_spread_by_product) > 6 else ""
        print(f"[INFO] sim spreads ({sim_spread_mode_norm}): {preview}{more}")

    # -- Historical depths (EXACT fill when possible).
    # Skip the expensive prices_to_order_depths build (~18s for 90k rows)
    # when EXACT mode is disabled (exact_days=0 or skip_hist_in_grid=True
    # with no final historical leg).
    if exact_days <= 0:
        historical_depths = {}
        ordered_keys_all = []
        ordered_keys = []
        have_exact = False
        print("[INFO] EXACT-fill: disabled (exact_days=0, skipping depth build)")
    else:
        historical_depths = prices_to_order_depths(prices_df)
        ordered_keys_all = sorted(historical_depths.keys())
        # By default restrict EXACT mode to ``exact_days`` days (much faster).
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
        print("[INFO] Using DefaultMarketMaker (no external trader specified).")
    else:
        print(f"[INFO] Using external trader: {trader_cls.__name__}")

    # Override inferred limits with trader-declared limits when provided.
    trader_limits = extract_trader_limits(trader_cls)
    if trader_limits:
        for prod, lim in trader_limits.items():
            position_limits[prod] = int(lim)
        print(f"[INFO] trader-declared position limits applied: {trader_limits}")

    # Market-trades index (for EXACT-mode traders that use state.market_trades).
    market_trades_index = build_market_trades_index(trades_df)
    if market_trades_index:
        n_products = sum(1 for _ in market_trades_index.values())
        print(f"[INFO] built market_trades index for {n_products} ticks")

    # -- Parameter registry.
    spec = extract_param_spec(trader_cls)
    if param_grid_overrides:
        spec = _apply_param_grid_overrides(spec, param_grid_overrides)
    print(f"[INFO] Parameter registry keys: {list(spec.keys())}")
    for _k, _v in spec.items():
        _grid = _v.get("grid", []) if isinstance(_v, dict) else []
        print(f"    {_k:<20s} type={_v.get('type','?'):<6s} n={len(_grid):<3d} grid={_grid}")

    # -- Monte-Carlo paths (truncated to max_ticks if set).
    # Per-product method resolution: base = global mc_method, overrides win.
    mc_method_overrides = dict(mc_method_overrides or {})
    all_products = list(prices_df["product"].unique())
    per_prod_method: Dict[str, str] = {
        p: mc_method_overrides.get(p, mc_method) for p in all_products
    }
    any_ou = any(m == "ou" for m in per_prod_method.values())
    any_bs = any(m == "bootstrap" for m in per_prod_method.values())
    if mc_method_overrides:
        print("[INFO] MC-method per product:")
        for p, m in per_prod_method.items():
            tag = " (override)" if p in mc_method_overrides else ""
            print(f"    {p:<30s} -> {m}{tag}")

    ou_params: Dict[str, Dict[str, float]] = {}
    mc_paths: Dict[str, np.ndarray] = {}
    if any_ou:
        print(f"[INFO] Generating {n_paths} Ornstein-Uhlenbeck paths ...")
        ou_paths_full, ou_params = generate_monte_carlo_paths_ou(
            prices_df, n_paths=n_paths, seed=seed,
            overrides=ou_overrides, horizon=max_ticks,
        )
        # Print a readable calibration table.
        print("[OU] per-product calibration & fit diagnostics:")
        print(f"    {'product':<28s} {'theta':>8s} {'mu':>12s} {'sigma':>10s} "
              f"{'half_life':>10s} {'r2_ar1':>8s} {'adf_like':>9s} "
              f"{'acf1':>7s} {'n_obs':>7s}")
        for p, fit in ou_params.items():
            print(f"    {p:<28s} {fit['theta']:>8.4f} {fit['mu']:>12.4f} "
                  f"{fit['sigma']:>10.4f} {fit['half_life']:>10.1f} "
                  f"{fit['r2_ar1']:>8.3f} {fit['adf_like']:>9.2f} "
                  f"{fit['mean_reversion_acf_decay']:>7.3f} {fit['n_obs']:>7d}")
        # Flag questionable fits to the user explicitly.
        for p, fit in ou_params.items():
            warnings_found = []
            if fit["adf_like"] < 2.0:
                warnings_found.append(f"adf_like={fit['adf_like']:.2f} < 2.0 (weak reversion -- OU may overfit)")
            if fit["r2_ar1"] < 0.01:
                warnings_found.append(f"r2_ar1={fit['r2_ar1']:.3f} < 0.01 (AR(1) explains almost nothing)")
            if fit["half_life"] > 10000:
                warnings_found.append(f"half_life={fit['half_life']:.0f} huge (effective random walk)")
            if abs(fit["resid_kurt"]) > 3:
                warnings_found.append(f"resid_kurt={fit['resid_kurt']:.1f} (heavy-tailed innovations, Gaussian OU will understate jumps)")
            if warnings_found:
                print(f"[OU WARN] {p}: " + "; ".join(warnings_found))
        # Persist the full calibration table for downstream inspection.
        cal_rows = []
        for p, fit in ou_params.items():
            row = dict(product=p)
            row.update({k: v for k, v in fit.items()})
            cal_rows.append(row)
        pd.DataFrame(cal_rows).to_csv(out_dir / "ou_calibration.csv", index=False)

    if any_bs:
        print(f"[INFO] Generating {n_paths} Monte-Carlo paths (bootstrap) ...")
        bs_paths_full = generate_monte_carlo_paths(prices_df, n_paths=n_paths, seed=seed)
    else:
        bs_paths_full = {}

    # Assemble final mc_paths by picking per-product method.
    for p in all_products:
        m = per_prod_method.get(p, mc_method)
        if m == "ou" and p in (ou_paths_full if any_ou else {}):
            mc_paths[p] = ou_paths_full[p]
        elif m == "bootstrap" and p in bs_paths_full:
            mc_paths[p] = bs_paths_full[p]
        elif any_ou and p in ou_paths_full:
            mc_paths[p] = ou_paths_full[p]
        elif p in bs_paths_full:
            mc_paths[p] = bs_paths_full[p]
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
            method=per_prod_method.get(p, mc_method),
        ))
    pd.DataFrame(mc_summary_rows).to_csv(out_dir / "monte_carlo_summary.csv", index=False)

    # OU validation plot (historical + simulated + terminal distribution).
    if any_ou and ou_params:
        ou_only_paths = {p: mc_paths[p] for p in ou_params.keys()
                         if per_prod_method.get(p, mc_method) == "ou" and p in mc_paths}
        if ou_only_paths:
            try:
                plot_ou_calibration(prices_df, ou_only_paths, ou_params,
                                    out_dir / "plot_ou_calibration.png")
            except Exception as e:
                print(f"[WARN] plot_ou_calibration failed: {e}")

    # -- Backtest-Runner (single param, over MC paths).
    T_path = min(arr.shape[1] for arr in mc_paths.values()) if mc_paths else 0
    hist_mids = {p: prices_df[prices_df["product"] == p]["mid_price"].values for p in products}

    # Trader path string -- passed to workers (path object is picklable but
    # the Trader class itself may not be; workers re-import by path).
    trader_path_str = str(trader_path) if trader_path else None

    # Shared-state initialiser args. These are pickled ONCE per worker
    # process instead of once per job -- a ~1000x reduction in pickle traffic
    # for the grid stage.
    worker_init_args = (
        trader_path_str,
        mc_paths,
        hist_mids if not skip_hist_in_grid else None,
        historical_depths if not skip_hist_in_grid else None,
        ordered_keys if not skip_hist_in_grid else None,
        position_limits,
        max_ticks,
        market_trades_index if not skip_hist_in_grid else None,
        n_paths,
        skip_hist_in_grid,
        risk_lambda,
        have_exact,
        seed,
        lean_metrics,
        sim_spread_by_product,
    )

    def _make_batch_runner(n_eval: int) -> Callable[[List[Dict[str, Any]]], pd.DataFrame]:
        n_eval = max(1, min(n_paths, n_eval))

        def run_batch(combos: List[Dict[str, Any]]) -> pd.DataFrame:
            # Per-job payload is now tiny: just (params, n_eval).
            jobs = [(c, n_eval) for c in combos]
            results = _parallel_map(
                jobs,
                _eval_combo_worker_shared,
                n_workers=n_workers,
                desc=f"grid (n_eval={n_eval})",
                initializer=_worker_init,
                initargs=worker_init_args,
            )
            rows = [r for r in results if r is not None]
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
            rank_metric=rank_metric, min_trades=min_trades_filter,
        )
    else:
        print(f"[INFO] Running single-stage grid search (eval_paths={eval_paths})")
        run_batch = _make_batch_runner(eval_paths)
        full_results = coarse_to_fine_grid(spec, run_batch, top_k=6)

    # --- Drop zombie rows: configs that never traded contribute zero signal
    # to any downstream analysis and only dilute the CSV. We keep one flat
    # 'active' frame for ranking + plots.
    full_results_full = full_results.copy()
    if "mean_trades_per_path" in full_results.columns:
        active_mask = full_results["mean_trades_per_path"] >= min_trades_filter
        n_drop = int((~active_mask).sum())
        if n_drop > 0:
            print(f"[INFO] dropping {n_drop} zero-/low-trade rows from output "
                  f"(mean_trades_per_path < {min_trades_filter})")
        full_results = full_results[active_mask].reset_index(drop=True)
    full_results.to_csv(out_dir / "full_grid_results.csv", index=False)
    print(f"[INFO] grid results (active): {len(full_results)} rows "
          f"(of {len(full_results_full)} total)")

    # ---------------------------------------------------------------
    # Stage-bias mitigation: coarse-stage combos were evaluated with
    # far fewer MC paths than fine-stage combos, so their sharpe
    # estimates have much higher variance. Ranking them side by side
    # with fine-stage combos lets a lucky coarse winner beat a truly
    # robust fine winner. When a two-stage CTF was run AND we have at
    # least some fine-stage rows, restrict the ranking pool to fine.
    #
    # Additionally, re-evaluate the top-K fine candidates with the
    # richer eval_paths_final budget before picking the winner. This
    # neutralises the extreme-value bias baked into nlargest().
    # Controlled by --skip-final-reval (on by default for robustness).
    # ---------------------------------------------------------------
    if ctf_enabled and "stage" in full_results.columns:
        fine_only = full_results[full_results["stage"] == "fine"].copy()
        n_coarse_in_top = 0
        if not fine_only.empty and rank_metric in fine_only.columns:
            # How many of the current Top-10 are from the coarse leg?
            cur_top10 = full_results.nlargest(10, rank_metric)
            n_coarse_in_top = int((cur_top10.get("stage", pd.Series()) == "coarse").sum())
            if n_coarse_in_top > 0:
                print(f"[RANK] stage-bias guard: removing {n_coarse_in_top} "
                      f"coarse-stage row(s) from Top-10 ranking pool "
                      f"(coarse was evaluated with n_eval={ctf_eval_coarse}, "
                      f"fine with n_eval={ctf_eval_fine} -- coarse stats are noisier)."
                      )
            ranking_pool = fine_only.reset_index(drop=True)
        else:
            ranking_pool = full_results
    else:
        ranking_pool = full_results

    # Final Top-K re-evaluation with the richer eval_paths_final budget.
    # Re-evaluates only the current Top-K candidates (default 10) so it
    # stays cheap -- K * eval_paths_final additional MC backtests.
    n_reval_eff = max(1, min(n_paths, eval_paths_final))
    reval_k = 10
    _median_n_eval = (
        float(ranking_pool["n_eval_paths"].median())
        if ("n_eval_paths" in ranking_pool.columns and not ranking_pool.empty)
        else 0.0
    )
    if np.isnan(_median_n_eval):
        _median_n_eval = 0.0
    if (not skip_final_reval
            and not ranking_pool.empty
            and rank_metric in ranking_pool.columns
            and n_reval_eff > int(_median_n_eval)):
        reval_cands = ranking_pool.nlargest(reval_k, rank_metric).reset_index(drop=True)
        reval_param_rows = [
            {k: r[k] for k in spec.keys() if k in r}
            for _, r in reval_cands.iterrows()
        ]
        print(f"[RANK] final re-evaluation: {len(reval_param_rows)} top candidates "
              f"x {n_reval_eff} MC paths each "
              f"(replaces noisy grid-stage sharpe with robust estimates)")
        reval_rows: List[Dict[str, Any]] = []
        reval_jobs = [(p, n_reval_eff) for p in reval_param_rows]
        reval_raw = _parallel_map(
            reval_jobs,
            _eval_combo_worker_shared,
            n_workers=n_workers, desc="final re-eval",
            initializer=_worker_init, initargs=worker_init_args,
        )
        for p, r in zip(reval_param_rows, reval_raw):
            if r is None:
                continue
            row = dict(p)
            row.update(r)
            row["stage"] = "reval"
            reval_rows.append(row)
        if reval_rows:
            reval_df = pd.DataFrame(reval_rows)
            # Merge reval rows back into ranking_pool. Drop the old grid
            # rows for the same param combos (identified by all spec keys)
            # so they don't double-count in downstream ranking.
            key_cols = list(spec.keys())
            rp = ranking_pool.copy()
            # Build composite key for matching
            rp["_reval_key"] = rp[key_cols].astype(str).agg("|".join, axis=1)
            reval_df["_reval_key"] = reval_df[key_cols].astype(str).agg("|".join, axis=1)
            rp = rp[~rp["_reval_key"].isin(reval_df["_reval_key"])]
            ranking_pool = pd.concat(
                [rp, reval_df], axis=0, ignore_index=True, sort=False
            ).drop(columns=["_reval_key"], errors="ignore").reset_index(drop=True)
            # Persist a dedicated CSV so the user can inspect re-eval deltas.
            reval_df.drop(columns=["_reval_key"], errors="ignore").to_csv(
                out_dir / "final_reval.csv", index=False,
            )
            print(f"[RANK] re-eval finished; merged {len(reval_df)} refined rows "
                  f"back into ranking pool. See final_reval.csv.")

    # -- Pareto + robust Top-2.
    pareto = pareto_frontier(full_results.dropna(subset=["variance", "mean_profit"]),
                             x_col="variance", y_col="mean_profit", maximize_y=True)
    pareto.to_csv(out_dir / "pareto_front.csv", index=False)

    # --- New artifact: Top-10 by Sharpe (submission-ready candidates).
    # Sourced from the (stage-filtered + re-eval'd) ranking_pool, not from
    # the raw full_results frame, to avoid stage-bias contamination.
    if "sharpe" in ranking_pool.columns and not ranking_pool.empty:
        top10_by_sharpe = ranking_pool.nlargest(10, "sharpe").reset_index(drop=True)
        top10_by_sharpe.to_csv(out_dir / "top10_by_sharpe.csv", index=False)
        print("[INFO] Top-10 by Sharpe (submission candidates):")
        show_cols = [c for c in [*spec.keys(), "mean_profit", "median_profit",
                                 "sharpe", "profit_per_trade", "mean_trades_per_path",
                                 "VaR_5", "CVaR_5", "n_eval_paths", "stage"]
                     if c in top10_by_sharpe.columns]
        print(top10_by_sharpe[show_cols].to_string(index=False))

    top2 = pick_top2_robust(ranking_pool, metric=rank_metric)
    top2.to_csv(out_dir / "top_parameter_pairs.csv", index=False)
    print(f"[INFO] Robust Top-2 parameter sets (ranked by '{rank_metric}'):")
    print(top2.to_string(index=False))

    # -- Visualization.
    # Historical equity for best params (one EXACT run, no grid overhead).
    # 'Best' is picked using the ranking metric from the ranking_pool
    # (stage-filtered + re-eval'd), so the fan chart and equity curve
    # reflect the robust winner rather than a coarse-stage lucky outlier.
    best_row = None
    if not ranking_pool.empty and rank_metric in ranking_pool.columns:
        best_row = ranking_pool.nlargest(1, rank_metric).iloc[0].to_dict()
        best_params = {k: best_row[k] for k in spec.keys() if k in best_row}
        print(f"[INFO] Best params by '{rank_metric}': {best_params} "
              f"(mean_profit={best_row.get('mean_profit', float('nan')):.1f}, "
              f"sharpe={best_row.get('sharpe', float('nan')):.3f})")
    elif not top2.empty:
        best_row = top2.iloc[0].to_dict()
        best_params = {k: best_row[k] for k in spec.keys() if k in best_row}
    else:
        best_params = None

    if best_params is not None:
        hist_res = run_backtest_on_series(
            trader_cls, best_params,
            mids_by_product=hist_mids,
            historical_depths=historical_depths if have_exact else None,
            ordered_keys=ordered_keys if have_exact else None,
            position_limits=position_limits,
            max_ticks=max_ticks,
            market_trades_index=market_trades_index if have_exact else None,
            sim_spread_by_product=sim_spread_by_product,
        )
        plot_equity_curve(hist_res["equity_curve"], out_dir / "plot_equity_curve.png")

        # Dump best params as JSON for easy reproducibility.
        import json as _json
        _br = best_row or {}
        (out_dir / "best_params.json").write_text(_json.dumps({
            "params": best_params,
            "rank_metric": rank_metric,
            "mean_profit": float(_br.get("mean_profit", float("nan"))),
            "sharpe": float(_br.get("sharpe", float("nan"))),
            "profit_per_trade": float(_br.get("profit_per_trade", float("nan"))),
            "mean_trades_per_path": float(_br.get("mean_trades_per_path", float("nan"))),
        }, indent=2, default=str))

        # Fan chart + final PnL distribution for the best params (parallel).
        fan_jobs = [
            (best_params, i) for i in range(min(eval_paths_final, n_paths))
        ]
        fan_results = _parallel_map(
            fan_jobs, _eval_single_path_worker_shared, n_workers=n_workers,
            desc="fan-chart", initializer=_worker_init, initargs=worker_init_args,
        )
        path_curves = [r["equity_curve"] for r in fan_results if r is not None]
        finals = [r["final_pnl"] for r in fan_results if r is not None]
        minlen = min(len(c) for c in path_curves) if path_curves else 0
        eq_matrix = np.array([c[:minlen] for c in path_curves])
        plot_fan_chart(eq_matrix, out_dir / "plot_mc_fan_chart.png")
        plot_final_pnl_dist(np.array(finals), out_dir / "plot_final_pnl_dist.png")

    # --- Core plots (always generated -- these are the submission essentials). ---
    plot_profit_vs_variance(full_results, out_dir / "plot_profit_vs_variance.png")
    plot_pareto(full_results, pareto, out_dir / "plot_pareto.png")
    plot_top_ranking(full_results, out_dir / "plot_top_ranking.png",
                     metric=rank_metric)

    # --- Extra diagnostic plots (skippable for smoke tests). ---
    if not skip_extra_plots:
        plot_sensitivity_math_low(full_results, out_dir / "plot_sensitivity_mathematical_low.png")
        plot_profit_hit_map(full_results, spec, out_dir / "plot_profit_hit_map.png",
                            metric=rank_metric)
        plot_parameter_sensitivity(full_results, spec,
                                   out_dir / "plot_parameter_sensitivity.png",
                                   metric=rank_metric)
        # Heatmap for the 2 most informative parameters (by column cardinality).
        num_params = [
            k for k, v in spec.items()
            if k in full_results.columns and pd.api.types.is_numeric_dtype(full_results[k])
        ]
        if len(num_params) >= 2:
            variances = [(k, full_results[k].nunique()) for k in num_params]
            variances.sort(key=lambda x: -x[1])
            k1, k2 = variances[0][0], variances[1][0]
            plot_heatmap_top2(full_results, (k1, k2), out_dir / f"plot_heatmap_{k1}_vs_{k2}.png")
    else:
        print("[INFO] skip_extra_plots=True -- skipping sensitivity/hit-map/heatmap plots")

    # --- Stability box plot (optional, ~100 extra backtests). ---
    if not skip_stability:
        _stab_metric = rank_metric if rank_metric in full_results.columns else "objective_score"
        top_k = full_results.nlargest(5, _stab_metric)
        stab_n = min(eval_paths_final // 2 if eval_paths_final >= 20 else 20, n_paths)
        stab_param_rows = [
            {k: r[k] for k in spec.keys() if k in r} for _, r in top_k.iterrows()
        ]
        labels = [
            "|".join(f"{k}={p[k]}" for k in list(p)[:2]) for p in stab_param_rows
        ]
        stab_jobs = [
            (p, i) for p in stab_param_rows for i in range(stab_n)
        ]
        stab_results = _parallel_map(
            stab_jobs, _eval_single_path_worker_shared, n_workers=n_workers,
            desc="stability", initializer=_worker_init, initargs=worker_init_args,
        )
        per_path_list = []
        for ci in range(len(stab_param_rows)):
            chunk = stab_results[ci * stab_n:(ci + 1) * stab_n]
            profits = np.array([r["final_pnl"] for r in chunk if r is not None])
            per_path_list.append(profits)
        plot_stability(per_path_list, labels, out_dir / "plot_stability_topK.png")
    else:
        print("[INFO] skip_stability=True -- skipping top-K stability box-plot stage")

    # Final FILL_MODE reflects the last backtest run. MC-path runs always use
    # APPROX (synthetic order books from simulated mids), so the last value
    # printed corresponds to the fan-chart/stability leg. The historical-EXACT
    # leg (if enabled via --grid-with-hist) uses real order books.
    print(f"[INFO] DONE in {time.time()-t0:.1f}s. Last fill mode: {FILL_MODE} "
          f"(MC paths always APPROX; historical leg uses EXACT when enabled).")
    print(f"[INFO] Artifacts written to: {out_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prosperity-4 MC/Grid Backtester (round-agnostic)")
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out-dir", type=Path, default=Path("./bt_out"))
    ap.add_argument("--trader", type=Path, default=None, help="Path to an external trader.py.")
    ap.add_argument("--round", dest="round_filter", type=int, default=None,
                    help="Optional filter for a specific round number (1, 2, 3, 4, 5).")
    ap.add_argument("--n-paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--risk-lambda", type=float, default=0.5,
                    help="Risk weight in objective_score = mean - lambda * variance.")
    ap.add_argument("--max-ticks", type=int, default=2000,
                    help="Cap ticks per backtest run (0 = no cap).")
    ap.add_argument("--exact-days", type=int, default=1,
                    help="Number of days used in EXACT mode (0 = disable EXACT).")
    ap.add_argument("--eval-paths", type=int, default=50,
                    help="MC paths per grid combination (speed / variance trade-off).")
    ap.add_argument("--eval-paths-final", type=int, default=200,
                    help="MC paths for the final best-param evaluation and fan chart.")
    ap.add_argument("--grid-with-hist", action="store_true",
                    help="Also run the historical EXACT backtest for every grid combo (slow).")
    ap.add_argument("--ctf", action="store_true",
                    help="Enable two-stage coarse-to-fine grid search.")
    ap.add_argument("--ctf-eval-coarse", type=int, default=10,
                    help="MC paths per combination in the coarse stage (default 10).")
    ap.add_argument("--ctf-eval-fine", type=int, default=100,
                    help="MC paths per combination in the fine stage (default 100).")
    ap.add_argument("--ctf-top-frac", type=float, default=0.10,
                    help="Fraction of the coarse stage used for refinement (default 0.10).")
    ap.add_argument("--ctf-n-interp", type=int, default=2,
                    help="Interpolation points between numeric top values in the fine stage (default 2).")
    ap.add_argument("--n-workers", type=int, default=-1,
                    help="Parallel worker processes (-1 = all CPU cores, 1 = single-threaded). "
                         "Requires joblib + tqdm (pip install joblib tqdm). Default: -1.")
    ap.add_argument("--rank-metric", type=str, default="sharpe",
                    choices=["sharpe", "mean_profit", "profit_per_trade",
                             "objective_score", "median_profit"],
                    help="Metric used to rank grid results (default: sharpe). "
                         "objective_score (mean - lambda*variance) is numerically "
                         "brittle when variance ~1e10; sharpe rewards consistency.")
    ap.add_argument("--min-trades-filter", type=float, default=5.0,
                    help="Drop configs with mean_trades_per_path < this value before "
                         "ranking and from the final output CSV. Default: 5.0.")
    ap.add_argument("--mc-method", choices=["bootstrap", "ou"], default="bootstrap",
                    help="Monte-Carlo generator: bootstrap (block+residual, default) "
                         "or ou (Ornstein-Uhlenbeck, mean-reverting parametric).")
    ap.add_argument("--ou-theta", type=float, default=None,
                    help="OU mean-reversion speed (global default for every product; "
                         "overridden by --ou-override).")
    ap.add_argument("--ou-mu", type=float, default=None,
                    help="OU long-run mean (global default for every product).")
    ap.add_argument("--ou-sigma", type=float, default=None,
                    help="OU volatility (global default for every product).")
    ap.add_argument("--ou-override", action="append", default=[],
                    help="Per-product OU override, format 'PRODUCT:theta=X,mu=Y,sigma=Z' "
                         "(any subset of keys). Repeatable.")
    ap.add_argument("--mc-method-override", action="append", default=[],
                    help="Per-product MC-method override, format 'PRODUCT=ou' or "
                         "'PRODUCT=bootstrap'. Repeatable. Overrides --mc-method "
                         "for the listed products only (e.g. OU for mean-reverting "
                         "products, bootstrap for trending/jumpy ones).")
    ap.add_argument("--position-limit", action="append", default=[],
                    help="Per-product position-limit override, format 'PRODUCT=N'. "
                         "Repeatable. Overrides the authoritative KNOWN_POSITION_LIMITS "
                         "table for the listed products only.")
    # Speed knobs -- skip optional diagnostics to shave wall-clock time.
    ap.add_argument("--skip-stability", action="store_true",
                    help="Skip the top-K stability box-plot stage (~100 extra backtests). "
                         "Recommended for smoke tests -- the fan chart already shows "
                         "per-path variance for the best params.")
    ap.add_argument("--skip-extra-plots", action="store_true",
                    help="Skip the heatmap, parameter-sensitivity and profit-hit-map "
                         "plots. Keeps the essentials (equity curve, fan chart, Pareto, "
                         "OU calibration, top-ranking, final PnL distribution).")
    ap.add_argument("--lean-metrics", action="store_true",
                    help="Skip per-path risk metrics (drawdown, hit-rate, turnover, "
                         "inventory-std, round-trips) during grid evaluation. "
                         "Keeps only final_pnl + n_trades -- faster per-combo loop.")
    ap.add_argument("--skip-final-reval", action="store_true",
                    help="Skip the final Top-K re-evaluation with eval_paths_final. "
                         "Default behaviour re-evaluates the Top-10 fine candidates "
                         "with the richer MC budget before picking the winner, which "
                         "removes the extreme-value bias from nlargest(). Disable only "
                         "for smoke tests -- it costs K * eval_paths_final extra runs.")
    ap.add_argument("--param-grid", action="append", default=[],
                    help="Override a trader parameter's grid with an arbitrary list "
                         "of thresholds. Format: 'KEY=v1,v2,v3[,...]'. Repeatable -- "
                         "supply once per parameter. Types are cast from the trader's "
                         "PARAM_SPEC (int/float/bool). Use this to test many "
                         "thresholds without editing the trader file, e.g. "
                         "--param-grid 'entry_sigma=0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0' "
                         "--param-grid 'window=20,40,60,80,100,150,200,300'.")
    # Volatility-smile analysis (VEV vouchers, Black-Scholes with r=0, q=0).
    ap.add_argument("--tte-base-days", type=int, default=DEFAULT_TTE_BASE_DAYS,
                    help="Calendar days until voucher expiry at day=0, timestamp=0 "
                         "(Prosperity-4 Round 3 dataset: 8). TTE_days = "
                         "tte_base_days - day - timestamp/1e6; T_years = "
                         "TTE_days/365 is the Black-Scholes input.")
    ap.add_argument("--skip-vol-smile", action="store_true",
                    help="Skip the VEV voucher volatility-smile diagnostics "
                         "(IV-vs-moneyness scatter, IV deviation TS, price "
                         "deviation TS). Auto-skipped when no VEV_* products "
                         "are in the dataset.")
    # Simulated bid-ask spread (APPROX-fill mode only). Repeatable -- accepts:
    #   AUTO            -> per-product mean of historical L1 spread (default).
    #   LEGACY          -> uniform 2.0 (the original floor(mid-1)/ceil(mid+1)).
    #   N               -> uniform float for every product (e.g. --sim-spread 6).
    #   PRODUCT=N       -> override one product (e.g. --sim-spread VEV_5500=4).
    # Multiple invocations are merged; per-product overrides win over a global N.
    ap.add_argument("--sim-spread", action="append", default=[],
                    help="APPROX-fill simulated bid-ask spread in ticks. "
                         "Accepts 'AUTO' (default: mean historical spread per "
                         "product), 'LEGACY' (uniform 2.0 -- pre-fix behaviour), "
                         "a global float (e.g. '--sim-spread 6'), or per-product "
                         "'PRODUCT=N' (repeatable, wins over the global value). "
                         "EXACT-fill mode ignores this flag entirely because it "
                         "already uses real historical L1 books.")
    # Volatility-smile strike groups. Repeatable -- format LABEL=K1,K2,...
    # Each group gets its own parabola fit, scatter PNG, and per-tick CSV in
    # addition to the baseline 'all' fit. Useful when ATM vouchers and the
    # wings live on different smiles (e.g. tick-floor pricing distorts deep
    # OTM IVs). Example:
    #   --smile-group core=5000,5100,5200,5300,5400,5500 \
    #   --smile-group wings=4000,4500,6000,6500
    ap.add_argument("--smile-group", action="append", default=[],
                    help="Per-group volatility-smile fit, format "
                         "'LABEL=K1,K2,...'. Repeatable. Each group produces "
                         "its own parabola (a, b, c), scatter plot "
                         "plot_vol_smile_scatter_<LABEL>.png, and per-tick CSV. "
                         "The baseline 'all' fit is always produced too.")
    return ap.parse_args()


def _parse_ou_overrides(global_theta: Optional[float],
                        global_mu: Optional[float],
                        global_sigma: Optional[float],
                        override_strings: List[str]) -> Dict[str, Dict[str, float]]:
    """Build OU override dict. Global values apply to every product key via the
    special '__GLOBAL__' bucket (consumed by calibrate_ou). Per-product strings
    (format 'PRODUCT:theta=X,mu=Y,sigma=Z') overlay those globals."""
    overrides: Dict[str, Dict[str, float]] = {}
    globals_dict: Dict[str, float] = {}
    if global_theta is not None:
        globals_dict["theta"] = float(global_theta)
    if global_mu is not None:
        globals_dict["mu"] = float(global_mu)
    if global_sigma is not None:
        globals_dict["sigma"] = float(global_sigma)
    if globals_dict:
        overrides["__GLOBAL__"] = globals_dict
    for raw in override_strings or []:
        if ":" not in raw:
            print(f"[WARN] Ignoring --ou-override '{raw}': missing ':' (expected PRODUCT:k=v,...)")
            continue
        product, kvs = raw.split(":", 1)
        product = product.strip()
        if not product:
            print(f"[WARN] Ignoring --ou-override '{raw}': empty product name")
            continue
        pdict: Dict[str, float] = {}
        for kv in kvs.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" not in kv:
                print(f"[WARN] Ignoring '{kv}' in --ou-override '{raw}' (expected k=v)")
                continue
            k, v = kv.split("=", 1)
            k = k.strip().lower()
            if k not in ("theta", "mu", "sigma"):
                print(f"[WARN] Ignoring unknown key '{k}' in --ou-override '{raw}'")
                continue
            try:
                pdict[k] = float(v)
            except ValueError:
                print(f"[WARN] Ignoring non-numeric '{kv}' in --ou-override '{raw}'")
        if pdict:
            overrides[product] = pdict
    return overrides


def _parse_kv_int_overrides(raw_list: List[str], label: str) -> Dict[str, int]:
    """Parse a repeatable 'PRODUCT=N' CLI flag into a {product: int} dict."""
    out: Dict[str, int] = {}
    for raw in raw_list or []:
        if "=" not in raw:
            print(f"[WARN] Ignoring --{label} '{raw}': missing '=' (expected PRODUCT=N)")
            continue
        prod, val = raw.split("=", 1)
        prod = prod.strip()
        if not prod:
            continue
        try:
            out[prod] = int(val.strip())
        except ValueError:
            print(f"[WARN] Ignoring --{label} '{raw}': '{val}' is not an integer")
    return out


def _parse_smile_groups(raw_list: List[str]) -> List[Tuple[str, List[int]]]:
    """Parse repeated ``--smile-group LABEL=K1,K2,...`` flags.

    Returns a list of (label, [strikes]) tuples, preserving CLI order.
    Invalid tokens are skipped with a warning.
    """
    groups: List[Tuple[str, List[int]]] = []
    for raw in raw_list:
        if "=" not in raw:
            print(f"[WARN] Ignoring --smile-group '{raw}': expected LABEL=K1,K2,...")
            continue
        label, vals = raw.split("=", 1)
        label = label.strip()
        strikes: List[int] = []
        for tok in vals.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                strikes.append(int(float(tok)))
            except ValueError:
                print(f"[WARN] Ignoring strike '{tok}' in --smile-group '{raw}': not numeric")
        if not label:
            print(f"[WARN] Ignoring --smile-group '{raw}': empty label")
            continue
        if not strikes:
            print(f"[WARN] Ignoring --smile-group '{raw}': no valid strikes")
            continue
        groups.append((label, strikes))
    return groups


def _parse_sim_spread(
    raw_list: List[str],
) -> Tuple[str, Dict[str, float]]:
    """Parse the repeatable ``--sim-spread`` flag into ``(mode, overrides)``.

    Mode is one of:
      * ``"auto"`` -- per-product mean historical spread (default).
      * ``"legacy"`` -- uniform :data:`DEFAULT_SIM_SPREAD` (2.0).
      * ``"global"`` -- a uniform float supplied as a bare positional value;
        the value lives in ``overrides["__GLOBAL__"]`` and is applied to every
        product after the auto/legacy seed.

    Per-product overrides (``PRODUCT=N``) are always merged into the dict and
    win over both the mode seed and any ``__GLOBAL__`` value.
    """
    mode = "auto"
    overrides: Dict[str, float] = {}
    for raw in raw_list or []:
        token = raw.strip()
        if not token:
            continue
        upper = token.upper()
        if upper in ("AUTO", "LEGACY"):
            mode = upper.lower()
            continue
        if "=" in token:
            prod, val = token.split("=", 1)
            prod = prod.strip()
            try:
                overrides[prod] = float(val.strip())
            except ValueError:
                print(f"[WARN] Ignoring --sim-spread '{raw}': '{val}' is not numeric")
            continue
        # Bare numeric value -> global uniform spread.
        try:
            overrides["__GLOBAL__"] = float(token)
        except ValueError:
            print(f"[WARN] Ignoring --sim-spread '{raw}': not a number, AUTO, LEGACY, or PRODUCT=N")
    # The '__GLOBAL__' sentinel (if present) is left in `overrides` and
    # expanded by run_pipeline across every product before per-product
    # overrides are merged on top.
    return mode, overrides


def _parse_kv_str_overrides(
    raw_list: List[str], label: str, allowed: List[str]
) -> Dict[str, str]:
    """Parse a repeatable 'PRODUCT=value' CLI flag into a {product: str} dict,
    validating value against ``allowed``."""
    out: Dict[str, str] = {}
    for raw in raw_list or []:
        if "=" not in raw:
            print(f"[WARN] Ignoring --{label} '{raw}': missing '=' (expected PRODUCT=value)")
            continue
        prod, val = raw.split("=", 1)
        prod = prod.strip()
        val = val.strip().lower()
        if not prod:
            continue
        if val not in allowed:
            print(f"[WARN] Ignoring --{label} '{raw}': value must be one of {allowed}")
            continue
        out[prod] = val
    return out


def _cast_param_value(raw: str, declared_type: Optional[str]) -> Any:
    """Cast a raw CLI token to the type declared in PARAM_SPEC.

    declared_type is one of {'int', 'float', 'bool', None}. Unknown / missing
    types fall back to best-effort (int -> float -> str). Bool accepts
    true/false/1/0/yes/no, case-insensitive.
    """
    s = raw.strip()
    t = (declared_type or "").lower()
    if t == "bool":
        low = s.lower()
        if low in ("true", "1", "yes", "y", "on"):
            return True
        if low in ("false", "0", "no", "n", "off"):
            return False
        raise ValueError(f"cannot parse bool from '{raw}'")
    if t == "int":
        # Accept '2.0' as 2 only when it's exactly integral.
        f = float(s)
        if not f.is_integer():
            raise ValueError(f"cannot cast non-integral '{raw}' to int")
        return int(f)
    if t == "float":
        return float(s)
    # Fallback: try int -> float -> str.
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_param_grid_overrides(raw_list: List[str]) -> Dict[str, List[str]]:
    """Parse a repeatable '--param-grid KEY=v1,v2,...' into {key: [raw_tokens]}.

    Values are kept as raw strings here; actual type casting happens in
    _apply_param_grid_overrides once the PARAM_SPEC type is known. Duplicate
    keys overwrite earlier definitions (last-one-wins), with a warning.
    """
    out: Dict[str, List[str]] = {}
    for raw in raw_list or []:
        if "=" not in raw:
            print(f"[WARN] Ignoring --param-grid '{raw}': missing '=' "
                  f"(expected KEY=v1,v2,...)")
            continue
        key, vals = raw.split("=", 1)
        key = key.strip()
        if not key:
            print(f"[WARN] Ignoring --param-grid '{raw}': empty key")
            continue
        tokens = [v.strip() for v in vals.split(",") if v.strip()]
        if not tokens:
            print(f"[WARN] Ignoring --param-grid '{raw}': no values after '='")
            continue
        if key in out:
            print(f"[WARN] --param-grid '{key}' specified twice; keeping last")
        out[key] = tokens
    return out


def _apply_param_grid_overrides(
    spec: Dict[str, Any], overrides: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Return a new PARAM_SPEC with overridden grids.

    - Unknown keys (not in spec) emit a warning and are dropped. We refuse to
      invent parameters, because the trader won't know about them.
    - Tokens are cast to the declared PARAM_SPEC type (int/float/bool). Tokens
      that fail to cast emit a warning and are skipped.
    - Duplicate values within a single grid are de-duplicated, preserving input
      order so users can control the coarse-to-fine scan direction.
    - Empty resulting grids fall back to the original PARAM_SPEC grid.
    """
    if not overrides:
        return spec
    new_spec: Dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v
                                for k, v in spec.items()}
    for key, raw_tokens in overrides.items():
        if key not in new_spec:
            print(f"[WARN] --param-grid '{key}' is not a known trader parameter "
                  f"(PARAM_SPEC keys: {list(new_spec.keys())}); skipping.")
            continue
        entry = new_spec[key]
        if not isinstance(entry, dict):
            print(f"[WARN] PARAM_SPEC['{key}'] is not a dict; cannot override.")
            continue
        declared_type = entry.get("type")
        cast_values: List[Any] = []
        seen: set = set()
        for tok in raw_tokens:
            try:
                val = _cast_param_value(tok, declared_type)
            except Exception as exc:
                print(f"[WARN] --param-grid '{key}': dropping '{tok}' ({exc})")
                continue
            # Use a hashable dedup key; for unhashable values (unlikely) fall
            # back to string form.
            try:
                dedup_key = val
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
            except TypeError:
                dedup_key = repr(val)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
            cast_values.append(val)
        if not cast_values:
            print(f"[WARN] --param-grid '{key}' produced no valid values; "
                  f"keeping PARAM_SPEC default {entry.get('grid')}")
            continue
        # Sort numeric grids so the coarse-to-fine interpolation logic works
        # as expected (it inspects sorted numeric neighbours).
        if declared_type in ("int", "float"):
            cast_values = sorted(cast_values)
        print(f"[PARAM-GRID] override '{key}': {cast_values} "
              f"(n={len(cast_values)}, type={declared_type or 'auto'})")
        entry["grid"] = cast_values
    return new_spec


if __name__ == "__main__":
    args = parse_args()
    _sim_mode, _sim_ovr = _parse_sim_spread(args.sim_spread)
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
            n_workers=args.n_workers,
            rank_metric=args.rank_metric,
            min_trades_filter=args.min_trades_filter,
            mc_method=args.mc_method,
            ou_overrides=_parse_ou_overrides(
                args.ou_theta, args.ou_mu, args.ou_sigma, args.ou_override,
            ),
            mc_method_overrides=_parse_kv_str_overrides(
                args.mc_method_override, "mc-method-override", ["ou", "bootstrap"],
            ),
            position_limit_overrides=_parse_kv_int_overrides(
                args.position_limit, "position-limit",
            ),
            skip_stability=args.skip_stability,
            skip_extra_plots=args.skip_extra_plots,
            lean_metrics=args.lean_metrics,
            skip_final_reval=args.skip_final_reval,
            param_grid_overrides=_parse_param_grid_overrides(args.param_grid),
            tte_base_days=args.tte_base_days,
            skip_vol_smile=args.skip_vol_smile,
            sim_spread_mode=_sim_mode,
            sim_spread_overrides=_sim_ovr,
            vol_smile_groups=_parse_smile_groups(args.smile_group),
        )
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
