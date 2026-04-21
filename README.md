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
- **Monte-Carlo path generator** layering block bootstrap → residual
  bootstrap → jump-aware perturbation.
- **Grid search** in two flavours:
  - single-stage (legacy `coarse_to_fine_grid` refinement),
  - explicit **two-stage coarse-to-fine** (`--ctf`): cheap noisy scan
    first, fine refinement only around the top fraction.
- **Robust selection**: Pareto frontier on *(variance, mean profit)*
  plus a neighbourhood-stability re-ranking. Returns the top-2 robust
  parameter sets.
- **Visualisation**: historical equity curve, MC fan chart, final-PnL
  distribution, variance/mean scatter, Pareto overlay, parameter
  heatmap, `mathematical_low` sensitivity plot, top-N ranking,
  top-K stability boxplot.
- **Zero-config fallback**: runs end-to-end against a synthetic 3-day,
  1-product dataset if no data is provided.

## Layout of the repository

```
backtesting-engine/
├── round3_backtest_montecarlo.py   # complete pipeline (≈1.8 kLoC, one file)
└── README.md                       # this file
```

The script is intentionally self-contained. Only `numpy`, `pandas`, and
`matplotlib` are required at runtime.

## Installation

```bash
git clone https://github.com/Diego-2510/backtesting-engine.git
cd backtesting-engine

python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

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
7. Monte-Carlo path generation.
8. Parameter registry + (two-stage) grid search.
9. Backtest core + metrics.
10. Pareto frontier + robust top-2 selection.
11. Visualisation.
12. Main pipeline.

## Performance tips

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

---

Sign conventions, product names, limits, and round layouts are all
sourced from `Prosperity.txt`. Architecture ideas — block structure of
`Trader.run`, EMA/linear-regression fair values, inventory skew, order
clipping — are inspired by the publicly available IMC Prosperity 3
second-place write-up (see `README.md` / `FrankfurtHedgehogs_polished.txt`
in the parent Space), but never treated as authoritative P4 facts.
