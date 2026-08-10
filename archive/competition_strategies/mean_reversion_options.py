"""Mean-reversion trader on option/fair-value spread — Prosperity 4 Round 3.

Strategy
--------
For each tradable voucher (call on VELVETFRUIT_EXTRACT), at every tick:

1. Pull the underlying mid S.
2. Extract the implied volatility of each voucher via Black-Scholes inversion.
3. Fit a parabola on the smile (IV as a function of moneyness).
4. Define the per-voucher fair value as BS(S, K, T, IV_smile_fit).
5. The "spread" is: spread(K) = mid(voucher) - fair(voucher).
   This quantity mean-reverts around 0 if the smile fit is a good
   estimator of the structural IV.
6. Enter SHORT when spread >= +entry_threshold (voucher too rich).
   Enter LONG  when spread <= -entry_threshold (voucher too cheap).
7. Exit when |spread| <= exit_threshold (mean reversion achieved),
   or after max_hold_ticks (safety timeout).

Independent per strike: one open round-trip at a time per voucher.

Conventions (Prosperity 4)
--------------------------
- Order.quantity > 0 = BUY, < 0 = SELL.
- OrderDepth.sell_orders values are NEGATIVE.
- traderData is the only persistence between iterations.

BSM dependency
--------------
This file tries to import the team library ``options_utils``. If
absent, a self-contained fallback using stdlib only (erf, bisection)
is used. Replace it by the team lib as soon as it is available.
Interface required (team contract):

    bs_call(S, K, T, sigma, r=0.0)           -> float
    implied_vol(price, S, K, T, r=0.0,
                sigma_lo=1e-4, sigma_hi=5.0) -> Optional[float]

Multi-day TTE handling
----------------------
Prosperity's ``TradingState`` does NOT expose ``state.day``. In a
live round the timestamp starts at 0 and grows monotonically within
a single trading day. In an offline multi-day backtest the timestamp
is RESET to ~0 at the start of each simulated day. To keep TTE
monotonically decreasing across both scenarios, we detect day
rollovers by observing ``state.timestamp < last_timestamp`` and
persist the cumulative day counter in ``traderData``.

TTE semantics (never mix):
  - ``tte_days_remaining`` = integer-ish calendar days until expiry
    (``tte_days - day_offset - fraction_of_day``)
  - ``T_years``            = ``tte_days_remaining / 365``  (BS input)
  - ``m_t``                = ``log(K / S) / sqrt(T_years)``  (moneyness)

TO CONFIRM before live deposit
------------------------------
1. ``tte_days`` at the start of round 3 (P4 statement). Submission
   spec from the engineer copilot: Round 3 live day = 5.
2. Position limit per voucher (hardcoded at 200, common P3 value).
3. Whether strike 5200 is reserved for the delta hedger (flip
   ``SKIP_5200_FOR_HEDGER`` accordingly).
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Any, Optional, Tuple
import json
import math


# =============================================================================
# Product metadata — TO UPDATE if P4 round 3 differs
# =============================================================================
UNDERLYING = "VELVETFRUIT_EXTRACT"

# Observed strikes in the CSVs:
# 4000, 4500: deep ITM, pure intrinsic (VEV_K = S - K). No IV edge. Excluded.
# 5000, 5100, 5200, 5300, 5400, 5500: tradable (smile-sensitive).
# 6000, 6500: floored at 0.5 (deep OTM). No signal. Excluded.
TRADABLE_STRIKES_ALL = [5000, 5100, 5200, 5300, 5400, 5500]

# Team coordination: if the delta hedger trades 5200, we skip it here.
# Flip to False if the hedger uses a different strike.
SKIP_5200_FOR_HEDGER = True

TRADABLE_STRIKES = (
    [k for k in TRADABLE_STRIKES_ALL if k != 5200]
    if SKIP_5200_FOR_HEDGER
    else list(TRADABLE_STRIKES_ALL)
)

VOUCHER_NAMES: Dict[int, str] = {k: f"VEV_{k}" for k in TRADABLE_STRIKES_ALL}


# =============================================================================
# BSM library import — fallback to stdlib implementation if not available
# =============================================================================
try:
    from options_utils import bs_call as _bs_call, implied_vol as _implied_vol  # type: ignore
    _HAS_TEAM_LIB = True
except Exception:  # pragma: no cover
    _HAS_TEAM_LIB = False

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
            return max(S - K * math.exp(-r * T), 0.0)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

    def _implied_vol(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float = 0.0,
        sigma_lo: float = 1e-4,
        sigma_hi: float = 5.0,
        tol: float = 1e-6,
        max_iter: int = 80,
    ) -> Optional[float]:
        """Bisection IV solver. Returns None on pathological inputs."""
        if T <= 0 or price <= 0:
            return None
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
        # Price below intrinsic is pathological. Price equal to intrinsic
        # implies zero vol which is not informative for the smile fit.
        if price < intrinsic - 1e-6 or price <= intrinsic + 1e-9:
            return None
        f_lo = _bs_call(S, K, T, sigma_lo, r) - price
        f_hi = _bs_call(S, K, T, sigma_hi, r) - price
        if f_lo * f_hi > 0.0:
            return None
        lo, hi = sigma_lo, sigma_hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = _bs_call(S, K, T, mid, r) - price
            if abs(f_mid) < tol:
                return mid
            if f_mid * f_lo < 0.0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)


# =============================================================================
# Smile fit (quadratic, stdlib only)
# =============================================================================
def _fit_parabola(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float, float]]:
    """Least-squares y = a + b*x + c*x^2 via normal equations.

    Returns (a, b, c) or None if the system is degenerate.
    """
    n = len(xs)
    if n < 3:
        return None
    sx = sum(xs)
    sx2 = sum(x * x for x in xs)
    sx3 = sum(x * x * x for x in xs)
    sx4 = sum(x * x * x * x for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2y = sum(x * x * y for x, y in zip(xs, ys))
    m = [
        [float(n), sx, sx2],
        [sx, sx2, sx3],
        [sx2, sx3, sx4],
    ]
    v = [sy, sxy, sx2y]

    def det3(mm):
        return (
            mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1])
            - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
            + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0])
        )

    def replace_col(mm, col_idx, vec):
        out = [row[:] for row in mm]
        for i in range(3):
            out[i][col_idx] = vec[i]
        return out

    d = det3(m)
    if abs(d) < 1e-12:
        return None
    a = det3(replace_col(m, 0, v)) / d
    b = det3(replace_col(m, 1, v)) / d
    c = det3(replace_col(m, 2, v)) / d
    return (a, b, c)


def _eval_parabola(coefs: Tuple[float, float, float], x: float) -> float:
    a, b, c = coefs
    return a + b * x + c * x * x


# =============================================================================
# Trader
# =============================================================================
class Trader:
    # --------------------- Prosperity interface hooks ------------------------
    LIMIT: Dict[str, int] = {
        # TO CONFIRM: replace 200 with the actual P4 round-3 limits
        f"VEV_{k}": 200 for k in TRADABLE_STRIKES_ALL
    }
    DEFAULT_LIMIT: int = 200

    # Grid-search registry (picked up by the backtester).
    # Keep the grids coarse; refine with --ctf --ctf-n-interp 4.
    PARAM_SPEC: Dict[str, Dict[str, Any]] = {
        "entry_threshold": {
            "type": "float",
            "grid": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        },
        "exit_threshold": {
            "type": "float",
            "grid": [0.1, 0.3, 0.5, 1.0, 1.5],
        },
        "max_hold_ticks": {
            "type": "int",
            "grid": [100, 250, 500, 1000, 2000],
        },
        "position_size": {
            "type": "int",
            "grid": [5, 10, 20, 40],
        },
        "tte_days": {
            # Kept in the grid only for sensitivity analysis; in live, set to
            # the official value from the P4 statement.
            "type": "float",
            "grid": [1.0, 2.0, 3.0, 5.0, 7.0],
        },
    }

    # Instance-level defaults (overwritten by the backtester via setattr
    # when a grid combination is evaluated).
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    max_hold_ticks: int = 250
    position_size: int = 10
    tte_days: float = 5.0  # live Round 3; backtests pass their own via setattr

    # --------------------- Internal constants --------------------------------
    RISK_FREE: float = 0.0
    MIN_VALID_IV: float = 1e-3
    MAX_VALID_IV: float = 3.0
    DAYS_PER_YEAR: float = 365.0
    TICKS_PER_DAY: int = 10_000  # Prosperity convention (timestamp 0..999_900)
    TICK_STEP: int = 100         # timestamp increment per tick
    MIN_SMILE_POINTS: int = 4    # parabola degenerate below this
    # Relative spread cap: |spread / mid| above this is treated as a
    # degenerate fit and the tick is skipped. 10% catches absurd signals
    # on low-priced OTM vouchers (e.g. VEV_5500 at mid ~7).
    MAX_ABS_SPREAD_PCT: float = 0.10

    # =========================================================================
    # Main entry point
    # =========================================================================
    def run(self, state: TradingState):
        td = self._load_td(state.traderData)

        # Detect day rollover across backtest sessions. Prosperity resets
        # ``state.timestamp`` at the start of each simulated day, so a
        # strictly-decreasing timestamp implies a new day. In a live
        # deposit this branch is never taken (single day, monotonic ts).
        last_ts = int(td.get("last_ts", -1))
        day_offset = int(td.get("day_offset", 0))
        if last_ts >= 0 and state.timestamp < last_ts:
            day_offset += 1
        td["day_offset"] = day_offset
        td["last_ts"] = int(state.timestamp)

        tick = int(td.get("tick", 0)) + 1
        td["tick"] = tick

        result: Dict[str, List[Order]] = {}

        # 1. Underlying mid
        if UNDERLYING not in state.order_depths:
            return result, 0, self._dump_td(td)
        S = self._mid(state.order_depths[UNDERLYING])
        if S is None or S <= 0:
            return result, 0, self._dump_td(td)

        # 2. Time to expiry in years (calendar days / 365, monotone across
        #    day rollovers thanks to ``day_offset``).
        T = self._time_to_expiry(state.timestamp, day_offset)
        if T <= 1e-9:
            # Near or past expiry: do nothing (positions should already be flat)
            return result, 0, self._dump_td(td)

        # 3. Collect IV per tradable voucher
        ivs: Dict[int, Tuple[float, float, int, int, OrderDepth]] = {}
        moneyness: Dict[int, float] = {}
        sqrt_T = math.sqrt(T)

        for K in TRADABLE_STRIKES:
            name = VOUCHER_NAMES[K]
            if name not in state.order_depths:
                continue
            depth = state.order_depths[name]
            mid_v = self._mid(depth)
            if mid_v is None or mid_v <= 0.6:
                # Skip floored (0.5) or malformed
                continue
            bb = max(depth.buy_orders) if depth.buy_orders else None
            ba = min(depth.sell_orders) if depth.sell_orders else None
            if bb is None or ba is None:
                continue
            iv = _implied_vol(
                mid_v,
                S,
                float(K),
                T,
                self.RISK_FREE,
                sigma_lo=self.MIN_VALID_IV,
                sigma_hi=self.MAX_VALID_IV,
            )
            if iv is None or not (self.MIN_VALID_IV < iv < self.MAX_VALID_IV):
                continue
            ivs[K] = (iv, mid_v, bb, ba, depth)
            moneyness[K] = math.log(K / S) / sqrt_T

        # 4. Need at least MIN_SMILE_POINTS to make the parabola non-degenerate
        if len(ivs) < self.MIN_SMILE_POINTS:
            return result, 0, self._dump_td(td)

        xs = [moneyness[K] for K in ivs]
        ys = [ivs[K][0] for K in ivs]
        coefs = _fit_parabola(xs, ys)
        if coefs is None:
            return result, 0, self._dump_td(td)

        # 5. For each tradable voucher, compute spread and decide
        entries: Dict[str, int] = td.setdefault("entries", {})

        for K, (iv_obs, mid_v, bb, ba, depth) in ivs.items():
            iv_hat = _eval_parabola(coefs, moneyness[K])
            if iv_hat <= 0.0:
                continue
            fair = _bs_call(S, float(K), T, iv_hat, self.RISK_FREE)
            spread = mid_v - fair

            # Sanity guard: absurd spreads come from degenerate fits.
            # Relative cap scales with voucher price so a 4-ticks spread on
            # VEV_5500 (mid~7) is flagged but 10 ticks on VEV_4000 (mid~1250)
            # is not.
            if mid_v > 0 and abs(spread) / mid_v > self.MAX_ABS_SPREAD_PCT:
                continue

            name = VOUCHER_NAMES[K]
            pos = int(state.position.get(name, 0))
            limit = int(self.LIMIT.get(name, self.DEFAULT_LIMIT))
            entry_tick = entries.get(name)

            orders = self._decide_voucher(
                name=name,
                spread=spread,
                pos=pos,
                limit=limit,
                best_bid=bb,
                best_ask=ba,
                tick=tick,
                entry_tick=entry_tick,
                depth=depth,
            )

            if orders:
                result[name] = orders
                # Maintain the entry-tick marker
                if pos == 0 and any(o.quantity != 0 for o in orders):
                    entries[name] = tick
                elif pos != 0 and self._flattens(orders, pos):
                    entries.pop(name, None)

        return result, 0, self._dump_td(td)

    # =========================================================================
    # Per-voucher decision logic
    # =========================================================================
    def _decide_voucher(
        self,
        name: str,
        spread: float,
        pos: int,
        limit: int,
        best_bid: int,
        best_ask: int,
        tick: int,
        entry_tick: Optional[int],
        depth: OrderDepth,
    ) -> List[Order]:
        orders: List[Order] = []
        entry = float(self.entry_threshold)
        exit_abs = float(self.exit_threshold)
        size = int(self.position_size)

        # 1. Forced flatten on timeout
        if pos != 0 and entry_tick is not None:
            held = tick - int(entry_tick)
            if held >= int(self.max_hold_ticks):
                return self._flatten(name, pos, best_bid, best_ask)

        # 2. Exit if mean-reverted (one-sided re-cross of the exit band)
        if pos > 0 and spread >= -exit_abs:
            return self._flatten(name, pos, best_bid, best_ask)
        if pos < 0 and spread <= exit_abs:
            return self._flatten(name, pos, best_bid, best_ask)

        # 3. Entries only if currently flat (one round-trip at a time)
        if pos == 0:
            if spread >= entry:
                # Voucher too rich -> SHORT: hit the best bid
                bid_vol = int(depth.buy_orders.get(best_bid, 0))
                qty = min(size, bid_vol, limit)
                if qty > 0:
                    orders.append(Order(name, best_bid, -qty))
            elif spread <= -entry:
                # Voucher too cheap -> LONG: hit the best ask
                ask_vol = int(abs(depth.sell_orders.get(best_ask, 0)))
                qty = min(size, ask_vol, limit)
                if qty > 0:
                    orders.append(Order(name, best_ask, qty))

        return orders

    # =========================================================================
    # Helpers
    # =========================================================================
    def _time_to_expiry(self, timestamp: int, day_offset: int = 0) -> float:
        """Convert (timestamp, day_offset) to T (years, the BS input).

        Model: every simulated day is 10000 ticks = 1_000_000 timestamp
        units (Prosperity convention). Inside that day, TTE linearly
        decreases by one unit. Across days, ``day_offset`` tracks how
        many full days have already elapsed (incremented by the caller
        on timestamp resets).

            tte_days_remaining = tte_days - day_offset - fraction_of_day
            T_years            = tte_days_remaining / 365

        In live Prosperity ``day_offset`` stays at 0 (single day) and
        the formula collapses to the original P3 convention.
        """
        ticks_this_day = timestamp // self.TICK_STEP  # 0..9999
        day_fraction = max(0.0, min(1.0, ticks_this_day / self.TICKS_PER_DAY))
        days_remaining = max(
            0.0,
            float(self.tte_days) - float(day_offset) - day_fraction,
        )
        return days_remaining / self.DAYS_PER_YEAR

    @staticmethod
    def _mid(depth: OrderDepth) -> Optional[float]:
        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @staticmethod
    def _flatten(name: str, pos: int, best_bid: int, best_ask: int) -> List[Order]:
        """Aggressive flatten: cross the spread for the full position."""
        if pos > 0:
            return [Order(name, best_bid, -pos)]
        if pos < 0:
            return [Order(name, best_ask, -pos)]  # -pos > 0 -> BUY
        return []

    @staticmethod
    def _flattens(orders: List[Order], pos: int) -> bool:
        """True iff the submitted orders fully close the current position."""
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
