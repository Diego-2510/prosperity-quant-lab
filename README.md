# Prosperity-4 Backtesting Engine

A round-agnostic backtesting, Monte-Carlo, and grid-search framework for
[IMC Prosperity 4](https://imc-prosperity.notion.site/prosperity-4-wiki)
Python traders.

Built around `round3_backtest_montecarlo.py`, the engine ingests official
Prosperity price and trade CSVs (or any compatible file layout), replays
them against an arbitrary `Trader` class, generates Monte-Carlo price
paths, optimises parameters with a coarse-to-fine grid search, and emits
a full set of artifacts (CSVs + plots) into an output directory.

---

## Table of contents

1. [Features](#features)
2. [Layout of the repository](#layout-of-the-repository)
3. [Installation](#installation)
4. [Data format](#data-format)
5. [Trader API](#trader-api)
6. [Quick start](#quick-start)
7. [CLI flags](#cli-flags)
8. [Output artifacts](#output-artifacts)
9. [How the pieces fit together](#how-the-pieces-fit-together)
10. [Performance tips](#performance-tips)
11. [Known limitations](#known-limitations)

---

## Features

- **File discovery** that recognises arbitrary Prosperity naming schemes
  (`prices_round_<R>_day_<D>*.csv`, `trades_round_<R>_day_<D>*.csv`) and
  falls back to generic CSVs with `{timestamp, product, mid_price}`.
- **Datamodel shim**: works whether or not the official
  [`datamodel`](https://github.com/imcolabs/prosperity) module is
  importable. Any Trader that does `from datamodel import …` is loaded
  out-of-the-box — the shim is injected into `sys.modules` when needed.
- **Prosperity-compatible fill engine** with a dual mode:
  - **EXACT**: real recorded L1/L2 order books, next-tick mid used as
    the reference for passive quote crossing.
  - **APPROX**: synthetic one-level book built from the mid series
    (`Bid = floor(mid-1)`, `Ask = ceil(mid+1)`) for fast MC.
- **Sign conventions** as per `Prosperity.txt`:
  - `OrderDepth.sell_orders` values are **negative**.
  - `Order.quantity > 0 = buy`, `Order.quantity < 0 = sell`.
  - Aggregated-side position-limit breach ⇒ *the entire side is
    dropped* for that product.
  - `traderData` is the only persistence between iterations.
- **Monte-Carlo path generator** with two selectable methods:
  - `bootstrap` (default): block bootstrap → residual bootstrap →
    jump-aware perturbation.
  - `ou`: Ornstein-Uhlenbeck mean-reverting parametric paths, with
    auto-calibration of `(θ, μ, σ)` per product (optionally overridable
    from the CLI).
- **Per-product MC-method overrides**: e.g. OU for mean-reverting
  underlyings, bootstrap for jumpy / trending products.
- **Black-Scholes + volatility-smile diagnostics** (Round 3 vouchers):
  automatic IV extraction (Newton-Raphson + bisection fallback),
  quadratic smile fit in `m_t = log(K/S)/sqrt(T_years)` space, and the
  three diagnostic plots used by the Frankfurt P3 top-2 README
  (IV-vs-moneyness scatter, IV deviation TS, price deviation TS).
  Assumes `r = 0.0`, no dividends, European calls. Auto-skipped when no
  `VEV_*` products are in the dataset.
- **Grid search** in two flavours:
  - single-stage (legacy `coarse_to_fine_grid` refinement),
  - explicit **two-stage coarse-to-fine** (`--ctf`): cheap noisy scan
    first, fine refinement only around the top fraction.
- **Top-K re-evaluation before winner selection**: the fine-stage
  top-10 candidates are re-run with `eval_paths_final` paths before
  naming a winner, removing the extreme-value bias of `nlargest()`.
  Disable with `--skip-final-reval` for smoke tests.
- **Robust selection**: Pareto frontier on *(variance, mean profit)*
  plus a neighbourhood-stability re-ranking. Returns the top-2 robust
  parameter sets.
- **Ad-hoc parameter-grid overrides** via `--param-grid 'KEY=v1,v2,...'`
  (repeatable), with automatic type casting from `PARAM_SPEC` — lets
  you test many thresholds without editing the trader file.
- **Visualisation**: historical equity curve, MC fan chart, final-PnL
  distribution, variance/mean scatter, Pareto overlay, parameter
  heatmap, `mathematical_low` sensitivity plot, top-N ranking,
  top-K stability boxplot.
- **Zero-config fallback**: runs end-to-end against a synthetic 3-day,
  1-product dataset if no data is provided.

## Layout of the repository

```
backtesting-engine/
├── round3_backtest_montecarlo.py   # complete pipeline (≈2 kLoC, one file)
├── mean_reversion.py               # sample z-score mean-reversion trader
├── requirements.txt                # pinned deps (incl. joblib + tqdm)
└── README.md                       # this file
```

The script is intentionally self-contained. `numpy`, `pandas`, and
`matplotlib` are strictly required at runtime; `joblib` and `tqdm` are
optional but strongly recommended (they unlock multi-core parallelism
and a live progress bar).

## Installation

```bash
git clone https://github.com/Diego-2510/prosperity-quant-lab.git
cd prosperity-quant-lab

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs everything, including the optional parallel-execution
stack (`joblib`, `tqdm`). If you prefer a minimal install, only
`numpy`, `pandas`, and `matplotlib` are strictly required — the engine
falls back to single-threaded execution and plain text progress when
`joblib`/`tqdm` are missing.

Python 3.10+ is required (3.12 is supported).

## Data format

### Price CSVs

Expected filename pattern (case-insensitive, with arbitrary trailing
upload suffixes):

```
prices_round_<R>_day_<D>[_anything].csv
```

Standard (official Prosperity) columns:

```
day; timestamp; product;
bid_price_1; bid_volume_1; bid_price_2; bid_volume_2; bid_price_3; bid_volume_3;
ask_price_1; ask_volume_1; ask_price_2; ask_volume_2; ask_price_3; ask_volume_3;
mid_price; profit_and_loss
```

Delimiter is auto-detected (`;` preferred, `,` fallback).

### Trade CSVs

```
trades_round_<R>_day_<D>[_anything].csv
```

Columns: `timestamp; buyer; seller; symbol; currency; price; quantity`
(plus an optional `day`). Trades are grouped by `(day, timestamp)` and
surfaced to the trader via `state.market_trades`.

### Generic fallback

Any CSV with at least `{timestamp, product, mid_price}` will be
accepted; a synthetic `bid/ask_price_1` is derived from the mid when
missing.

## Trader API

Drop a file exposing a `class Trader` with a `run(self, state)` method:

```python
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    # Optional — used by the engine to override inferred position limits:
    LIMIT = {"PRODUCT_A": 50, "PRODUCT_B": 80}

    # Optional — if present, the keys become grid-search parameters:
    PARAM_SPEC = {
        "threshold": {"type": "int", "grid": [2, 3, 4]},
        "aggressive": {"type": "bool", "grid": [False, True]},
    }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        # ... your logic ...
        trader_data = ""
        conversions = 0
        return result, conversions, trader_data
```

The loader (`load_external_trader`) uses `importlib.util`, so files with
hyphens in the name (e.g. `auto-trading.py`) work fine. If the official
`datamodel` module is not installed, a local shim is injected before
the module is imported.

### Parameter discovery

Grid search parameters are discovered in this order:

1. `Trader.PARAM_SPEC` — preferred. Each entry is
   `{"type": "int"|"float"|"bool"|"str", "grid": [...]}`.
2. `Trader.CONFIG` (dict of numeric / bool values) — small generic
   grids are synthesised around each numeric value.
3. Otherwise the spec is empty and the engine runs a single backtest
   with no parameter tuning (useful for compatibility testing).

### Position limits

Resolved in this order (last wins):

1. Inferred from the largest observed per-snapshot volume, floored at
   `DEFAULT_POSITION_LIMIT = 50`.
2. Overridden per-product by `Trader.LIMIT` (or `LIMITS`,
   `POSITION_LIMIT`, `POSITION_LIMITS`) when declared.

## Quick start

Minimal run against the bundled data layout:

```bash
python round3_backtest_montecarlo.py \
    --data-dir ./data \
    --out-dir  ./bt_out \
    --trader   ./my_trader.py \
    --n-paths  1000 \
    --seed     42
```

Fast compatibility smoke test (e.g. for `auto-trading.py` that has no
`PARAM_SPEC`):

```bash
python round3_backtest_montecarlo.py \
    --data-dir ./data_r1 \
    --out-dir  ./bt_out_autotrading \
    --trader   ./auto-trading.py \
    --n-paths  20 --max-ticks 300 \
    --eval-paths 10 --eval-paths-final 20 \
    --exact-days 1
```

Two-stage coarse-to-fine grid search:

```bash
python round3_backtest_montecarlo.py \
    --data-dir ./data \
    --out-dir  ./bt_out_ctf \
    --trader   ./my_trader.py \
    --ctf \
    --ctf-eval-coarse 10 \
    --ctf-eval-fine   100 \
    --ctf-top-frac    0.10 \
    --ctf-n-interp    0       # tight refinement on observed top values only
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | `./data` | Directory (or file) with Prosperity CSVs. |
| `--out-dir PATH` | `./bt_out` | Destination for all generated artifacts. |
| `--trader PATH` | *(none)* | Path to an external `trader.py`. |
| `--round N` | *(none)* | Only load files from round `N`. |
| `--n-paths N` | `1000` | Total Monte-Carlo paths to generate. |
| `--seed N` | `42` | RNG seed. |
| `--risk-lambda F` | `0.5` | Weight in `objective_score = mean − λ·variance`. |
| `--max-ticks N` | `2000` | Cap ticks per backtest run (`0` disables cap). |
| `--exact-days N` | `1` | Days used in EXACT mode (`0` disables EXACT). |
| `--eval-paths N` | `50` | MC paths per grid combo (speed / variance trade-off). |
| `--eval-paths-final N` | `200` | MC paths for best-param evaluation + fan chart. |
| `--grid-with-hist` | *off* | Also run the historical EXACT pass for every grid combo (slow). |
| `--ctf` | *off* | Enable two-stage coarse-to-fine grid search. |
| `--ctf-eval-coarse N` | `10` | MC paths per combo in the coarse stage. |
| `--ctf-eval-fine N` | `100` | MC paths per combo in the fine stage. |
| `--ctf-top-frac F` | `0.10` | Top fraction of the coarse stage used for refinement. |
| `--ctf-n-interp N` | `2` | Interpolation points inserted between numeric top values. |
| `--n-workers N` | `-1` | Parallel worker processes (`-1` = all CPU cores, `1` = single-threaded). Requires `joblib` + `tqdm`. |
| `--rank-metric METRIC` | `sharpe` | Ranking metric for grid results. One of `sharpe`, `mean_profit`, `profit_per_trade`, `objective_score`, `median_profit`. `sharpe` rewards consistency; `objective_score` (mean − λ·variance) is numerically brittle when variance is large. |
| `--min-trades-filter F` | `5.0` | Drop grid configs whose `mean_trades_per_path` is below this value before ranking. |
| `--mc-method {bootstrap,ou}` | `bootstrap` | Monte-Carlo generator. `ou` produces Ornstein-Uhlenbeck mean-reverting paths (ideal for structurally mean-reverting products). |
| `--ou-theta F` / `--ou-mu F` / `--ou-sigma F` | auto | Global OU parameter defaults (mean-reversion speed, long-run mean, volatility). Any subset can be passed; missing keys fall back to per-product calibration. |
| `--ou-override 'PROD:theta=X,mu=Y,sigma=Z'` | *(repeatable)* | Per-product OU override (any subset of keys). Overlays the globals. |
| `--mc-method-override 'PROD=ou'` | *(repeatable)* | Per-product MC-method override (e.g. OU for mean-reverting products, bootstrap for trending/jumpy ones). |
| `--position-limit 'PROD=N'` | *(repeatable)* | Per-product position-limit override; overrides the authoritative `KNOWN_POSITION_LIMITS` table for the listed products only. |
| `--skip-stability` | *off* | Skip the top-K stability box-plot stage (~100 extra backtests). |
| `--skip-extra-plots` | *off* | Skip heatmap / sensitivity / hit-map plots. Keeps equity curve, fan chart, Pareto, top-ranking, OU calibration, final-PnL distribution. |
| `--lean-metrics` | *off* | Skip per-path drawdown / hit-rate / turnover / inventory-std / round-trips during grid evaluation (keeps only `final_pnl` + `n_trades`). |
| `--skip-final-reval` | *off* | Skip the final top-K re-evaluation with `eval_paths_final`. |
| `--param-grid 'KEY=v1,v2,...'` | *(repeatable)* | Override a trader parameter's grid with an arbitrary list. Types are cast from `PARAM_SPEC`. Use to test many thresholds without editing the trader file. Example: `--param-grid 'entry_sigma=0.75,1.0,1.25,1.5,2.0' --param-grid 'window=20,40,60,80,100'`. |
| `--tte-base-days N` | `8` | Calendar days until voucher expiry at `day=0, timestamp=0` (Prosperity-4 Round 3 dataset = 8). The BS input is `T_years = (tte_base_days − day − timestamp/1e6) / 365`. |
| `--skip-vol-smile` | *off* | Skip the VEV voucher volatility-smile diagnostics. Auto-skipped when no `VEV_*` / `VELVETFRUIT_EXTRACT_VOUCHER_*` products are in the dataset. |

## Output artifacts

Everything lands in `--out-dir`:

| File | Contents |
|------|----------|
| `feature_diagnostics.csv` | Per-product volatility, jump rate, regime stats. |
| `monte_carlo_summary.csv` | MC path count, horizon, mean/std of final prices. |
| `full_grid_results.csv` | Metrics for every evaluated parameter combo. |
| `pareto_front.csv` | Pareto-optimal combos on *(variance, mean profit)*. |
| `top_parameter_pairs.csv` | Robust top-2 parameter sets. |
| `plot_equity_curve.png` | Best-param historical equity (EXACT run). |
| `plot_mc_fan_chart.png` | Monte-Carlo fan chart for best params. |
| `plot_final_pnl_dist.png` | Histogram of final PnL across MC paths. |
| `plot_profit_vs_variance.png` | Scatter over all grid combos. |
| `plot_pareto.png` | Pareto frontier overlay. |
| `plot_top_ranking.png` | Top-20 combos by `objective_score`. |
| `plot_heatmap_<k1>_vs_<k2>.png` | Parameter heatmap for the 2 most informative keys. |
| `plot_sensitivity_mathematical_low.png` | Sensitivity on the `mathematical_low` floor. |
| `plot_stability_topK.png` | Distribution of final PnL per MC path for the top-5. |
| `plot_ou_calibration.png` | Per-product OU fit diagnostics (only when `--mc-method ou` or an `--ou-override` is active). |
| `vol_smile_per_tick.csv` | Per-(day, timestamp, strike) row with market IV, moneyness `m_t`, smile IV, theoretical BS price, IV deviation, price deviation. Only when VEV products are present. |
| `vol_smile_fit.json` | Parabola coefficients `[a, b, c]` for `IV_hat(m_t) = a·m² + b·m + c`, plus the BS assumptions (`r=0`, no dividends), fit size, and ranges. |
| `plot_vol_smile_scatter.png` | IV vs moneyness scatter with fitted parabola, one colour per strike (figure 6a). |
| `plot_iv_deviation_ts.png` | Time series of `market_IV − smile_IV` per strike (figure 6b). |
| `plot_price_deviation_ts.png` | Time series of `market_price − BS_theo(smile_IV)` per strike (figure 6c). |

## How the pieces fit together

```text
                ┌──────────────────┐
                │ Prosperity CSVs  │
                └────────┬─────────┘
                         │ load_prices / load_trades
                         ▼
                ┌──────────────────┐
                │ feature frame    │
                └──┬────────────┬──┘
                   │            │
         historical order-books │
                   │            │ Monte-Carlo paths
                   ▼            ▼
      ┌────────────────────┐   ┌────────────────────────┐
      │ EXACT backtest     │   │ APPROX backtest        │
      │ (real L1/L2)       │   │ (synthetic book @ mid) │
      └────────┬───────────┘   └────────────┬───────────┘
               │                            │
               └────────────┬───────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ grid / two-stage    │
                 │ coarse-to-fine      │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Pareto + robust     │
                 │ top-2 selection     │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ CSVs + PNGs         │
                 └─────────────────────┘
```

Sections in the script (numbered comments at the top of each):

0. Global config and assumptions.
1. Datamodel compatibility layer (local shim if `datamodel` missing).
2. File discovery and parsing.
3. Order-book normalisation + mid / micro-price.
4. Prosperity-compatible fill engine.
5. Trader wrapper (external + default market maker).
6. Feature engineering + diagnostics.
7. Monte-Carlo path generation (bootstrap + OU).
8. Parameter registry + (two-stage) grid search + `--param-grid` overrides.
9. Backtest core + metrics (EXACT + APPROX with round-trip PnL attribution).
10. Pareto frontier + robust top-2 selection + final top-K re-evaluation.
11. Black-Scholes + volatility-smile diagnostics (Round 3 vouchers).
12. Visualisation.
13. Main pipeline.

## Performance tips

- **Use all your cores.** By default `--n-workers -1` dispatches the
  grid search, fan chart, and stability evaluation to a `joblib` /
  `loky` process pool — one worker per CPU core. A live `tqdm`
  progress bar shows throughput. On an 8-core laptop expect roughly a
  6–7× wall-clock speedup over single-threaded execution; on a 16-core
  box, 10–14×. Pass `--n-workers 1` to restore the legacy
  single-threaded behaviour for debugging.
- Because workers are separate OS processes, your laptop CPU should be
  pinned near 100 % during the grid stage. If it is not, check whether
  `joblib` and `tqdm` are actually installed (`pip install joblib
  tqdm`) — the engine warns and falls back to serial execution when
  they are missing.
- `--max-ticks` is the single most effective knob during iteration. A
  value of 300–600 keeps a full pipeline under a minute even on modest
  hardware.
- Keep `--exact-days 1` during parameter sweeps; bump to 3 only for the
  final report.
- Use `--ctf --ctf-n-interp 0` when your numeric grid is already dense
  — the fine stage then re-evaluates only the top subset with higher
  MC budget, without blowing up the combination count.
- The historical EXACT pass is expensive. It is skipped inside the
  grid search by default; pass `--grid-with-hist` to re-enable it.
- **Three opt-in speed knobs** for smoke tests: `--skip-stability`
  (skips the top-K box-plot leg, ~100 extra backtests), `--skip-extra-plots`
  (skips sensitivity / hit-map / heatmap, keeps equity curve + fan chart +
  Pareto + top-ranking + OU calibration + final-PnL distribution),
  and `--lean-metrics` (skips per-path drawdown / hit-rate / turnover /
  inventory-std — keeps only `final_pnl` and `n_trades` for ranking).

### Sizing the CTF grid — avoid multi-hour Stage-2 runs

The fine stage grid size grows **geometrically** with `--ctf-n-interp`
across the number of numeric parameters.  For a 4-parameter PARAM_SPEC
(typical `mean_reversion.py`: `entry_sigma, sigma_gap, max_hold_ticks,
window`), empirical Stage-2 sizes on a top-40 coarse selection
(`--ctf-top-frac 0.15` of a 400-combo coarse grid) are:

| `--ctf-n-interp` | Stage-2 combos | Ballpark wall time (16 cores, `n-eval=15`) |
|:---:|---:|---:|
| 0 | ~40    | ~15 s |
| 1 | ~1 400 | ~10 min |
| 2 | ~7 000 | ~50 min |
| 3 | ~25 000 | ~3 h 30 min |
| 4 | ~65 000 | ~8 h |

Rule of thumb: **start with `--ctf-n-interp 1`** for daily iteration,
bump to `2` only when a parameter is clearly under-resolved, and reserve
`3` for the very last submission-candidate sweep. A full Round-3 baseline
with 400 coarse × 1 400 fine combos, 10 products simulated, `n-eval=15`,
and `--n-workers -1` finishes under 15 minutes on a 16-core laptop.

## Black-Scholes + volatility-smile diagnostics

When the dataset contains `VEV_*` or `VELVETFRUIT_EXTRACT_VOUCHER_*`
products, the pipeline automatically runs a dedicated Round-3 diagnostic
block before parameter search. It extracts an implied-volatility surface
from the voucher mids, fits a quadratic smile in moneyness space, and
writes the five artefacts listed above.

### Assumptions (per user spec)

- `r = 0.0` (annualized, continuously compounded risk-free rate).
- `q = 0.0` (no dividends).
- European calls, no early exercise.
- **TTE semantics are strictly separated** — never mix the two:
  - `TTE_days = tte_base_days − day − timestamp/1e6` (calendar days).
  - `T_years = TTE_days / 365` (the Black-Scholes input).
  - `m_t    = log(K / S) / sqrt(T_years)` (moneyness).
- Only voucher rows with `extrinsic = market_price − max(S−K, 0) > 0.5`
  enter the smile fit. This drops pure-intrinsic deep-ITM rows
  (VEV_4000/4500 at mid ≈ S−K) and floored deep-OTM rows
  (VEV_6000/6500 at mid ≈ 0.50).

### IV solver

Newton-Raphson in `(1e-4, 5.0)` with bisection fallback, minimum 10
valid points required for the degree-2 polyfit. Solver returns `None`
on pathological prices (below intrinsic or above the high-σ bound) and
those rows are dropped before the fit.

### Reusing the fit in a trader

The parabola coefficients (`[a, b, c]`, highest-degree-first) are
written to `vol_smile_fit.json`. A Prosperity trader can reuse them
as structural IV estimates:

```python
# inside trader.run()
m_t   = math.log(K / S) / math.sqrt(T_years)
iv_hat = a * m_t * m_t + b * m_t + c      # smile IV for this K
fair   = bs_call(S, K, T_years, iv_hat)    # theoretical BS price
spread = market_price - fair               # mean-reverting around 0
```

This is exactly the contract used by `mean_reversion_options.py`
(Round 3 mean-reverter on the option/fair spread).

## Known limitations

- **No conversion mechanics**. The engine ignores the `conversions`
  return value. Conversion-reliant strategies (Round 5-style) need a
  follow-up extension.
- **Observations are empty**. `state.observations` is populated with
  empty `plainValueObservations` / `conversionObservations` dicts.
- **APPROX mode uses a one-level synthetic book**, so traders that
  depend on deep-book shape behave differently in APPROX vs EXACT.
- **`traderData` is opaque**: the engine does not inspect, only
  round-trips it across ticks.
- **Parameter discovery is best-effort**: heuristic `CONFIG`-dict
  support is intentionally conservative; prefer `PARAM_SPEC`.

