# Prosperity Quant Lab

> **IMC Prosperity 4 — Competition complete.**  
> **#326 globally · #200 Algo · #10 in the country · 30,703 participants**

A full-cycle quantitative research and trading systems project built for the [IMC Prosperity 4](https://imc-prosperity.notion.site/prosperity-4-wiki) algorithmic trading competition. Starting from raw market data and competition constraints, this project delivers a production-grade backtesting engine, systematic parameter optimisation, and options pricing infrastructure that was used to compete against 30,703 participants worldwide.

---

## Results

| Metric | Result |
|---|---|
| Global rank | **#326** |
| Algo rank | **#200** |
| Country rank | **#10** |
| Total participants | **30,703** |

---

## Project Overview

Most competition repositories are one-off scripts. This one is a **research platform**. The codebase was designed so that any new strategy can be plugged in, automatically grid-searched across thousands of parameter combinations, evaluated on Monte-Carlo simulated paths, and validated against real competition data — all from a single CLI command.

Key technical areas covered:

- Event-driven backtesting engine replicating Prosperity exchange mechanics exactly
- Monte-Carlo simulation with block bootstrap and Ornstein-Uhlenbeck models
- Coarse-to-fine grid search with Pareto-optimal parameter selection
- Black-Scholes implied volatility surface fitting for Round 3 options products
- Parallel execution via joblib achieving 10-14x speedup on multi-core hardware

---

## Repository Structure

```
prosperity-quant-lab/
├── backtest_montecarlo.py      # Core engine: backtest, MC simulation, grid search, BS analytics
├── mean_reversion.py           # Primary competition strategy (z-score market maker)
├── mean_reversion_options.py   # Options strategy using fitted smile IV
├── trader_multistrike.py       # Round-specific multi-strike trader
├── bt_out_submission/          # Generated output artifacts (plots, JSON)
├── ROUND_3_old/                # Legacy Round 3 strategy files
├── requirements.txt            # Pinned dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 to 3.12 (3.12 recommended)
- `pip` or a virtual environment manager (`venv`, `conda`)
- Competition log CSVs exported from the [Prosperity platform](https://imc-prosperity.notion.site/prosperity-4-wiki) (price history and trade history per round)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Diego-2510/prosperity-quant-lab.git
cd prosperity-quant-lab

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `joblib`, `tqdm`, and `scipy` are listed as optional in `requirements.txt` but are strongly recommended. Without `joblib`, the grid search runs single-threaded. Without `scipy`, IV inversion falls back to a hand-rolled `erf` implementation (roughly 5x slower).

### Preparing Competition Data

The backtester expects Prosperity CSV log files, one per round. These can be downloaded from your competition dashboard after each round ends. Place them in a dedicated directory:

```
data/
├── round3_prices.csv
├── round3_trades.csv
└── ...
```

The expected column schema matches Prosperity's native export format (`timestamp`, `product`, `bid_price_1`, `ask_price_1`, `mid_price`, etc.).

---

## Usage

### Running the Backtester

```bash
python backtest_montecarlo.py \
  --prices data/round3_prices.csv \
  --trades data/round3_trades.csv \
  --strategy mean_reversion \
  --n-workers -1
```

| Flag | Description | Default |
|---|---|---|
| `--prices` | Path to Prosperity prices CSV | required |
| `--trades` | Path to Prosperity trades CSV | required |
| `--strategy` | Strategy module name (without `.py`) | required |
| `--n-workers` | Parallel workers (`-1` = all cores) | `1` |
| `--param-grid` | Override parameter grid as JSON string | uses `PARAM_SPEC` in strategy |
| `--fill-mode` | `EXACT` (real L1/L2 books) or `APPROX` (synthetic mid) | `EXACT` |
| `--mc-paths` | Number of Monte-Carlo paths per evaluation | `500` |
| `--top-k` | Re-evaluate top-K configs to remove selection bias | `10` |
| `--output-dir` | Directory for plots and JSON artifacts | `bt_out_submission/` |

### Full Grid Search Example

```bash
python backtest_montecarlo.py \
  --prices data/round3_prices.csv \
  --trades data/round3_trades.csv \
  --strategy mean_reversion \
  --n-workers -1 \
  --mc-paths 1000 \
  --top-k 20 \
  --output-dir results/round3/
```

### Overriding Parameters Without Editing Source Files

Any parameter defined in a strategy's `PARAM_SPEC` can be overridden directly from the CLI:

```bash
python backtest_montecarlo.py \
  --prices data/round3_prices.csv \
  --trades data/round3_trades.csv \
  --strategy mean_reversion \
  --param-grid '{"window": [20, 50, 100], "z_entry": [1.5, 2.0, 2.5]}'
```

### Running Individual Strategies

```bash
python mean_reversion.py
python mean_reversion_options.py
```

### Output Artifacts

After a successful run, the output directory will contain:

```
results/round3/
├── equity_curves.png       # PnL over time per configuration
├── mc_fan_chart.png        # Monte-Carlo path envelope
├── pareto_overlay.png      # Mean vs. variance Pareto frontier
├── heatmap_<param>.png     # Sensitivity heatmaps per parameter pair
├── stability_boxplots.png  # Out-of-sample neighbourhood stability
├── vol_smile_fit.json      # Fitted smile coefficients (Round 3 options)
└── best_params.json        # Recommended parameters for live submission
```

The `vol_smile_fit.json` file is directly reusable in the live trader: import the fitted parabola coefficients into `mean_reversion_options.py` before submitting to get accurate fair-value estimates for options products.

---

## Engineering Highlights

**Modular strategy interface.** Any `Trader` class that exposes a `PARAM_SPEC` dictionary is automatically grid-searched with no manual wiring. Parameter types are auto-cast from the JSON grid definition, and the entire grid can be overridden at CLI level without touching the strategy file. This enabled sub-minute iteration cycles during a live competition where time is the binding constraint.

**Two-stage optimisation to prevent overfitting.** A cheap coarse search narrows the parameter space, followed by a high-fidelity fine search on the survivors. The final winner is selected by Pareto frontier on `(mean PnL, variance)` rather than raw peak PnL, and then re-validated with a separate top-K pass to remove lucky-draw bias from `nlargest()`.

**Options pricing as first-class infrastructure.** For Round 3 voucher products, the pipeline extracts a full implied-volatility surface using Newton-Raphson inversion, fits a quadratic moneyness smile, and serialises the coefficients to `vol_smile_fit.json`. The live trader reads this file directly, making the analytics pipeline and the submission pipeline tightly coupled by design.

---

## Key Lessons

1. **Infrastructure compounds.** The time invested in a robust backtesting loop in Round 1 paid back across every subsequent round as a fast, trusted feedback mechanism.
2. **Robustness beats peak performance.** Selecting parameters by Pareto frontier on mean and variance rather than raw PnL consistently outperformed greedy top-1 selection in out-of-sample evaluation.
3. **Derivatives require a pricing model, not a heuristic.** Moving from ad-hoc voucher pricing to a proper IV surface fit produced measurably better edge estimation for the Round 3 options market.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Numerical core | NumPy, Pandas |
| Visualisation | Matplotlib |
| Parallelism | joblib, tqdm |
| Derivatives pricing | Custom Black-Scholes engine (European calls, r=0) |
| Optimisation | Grid search with coarse-to-fine Pareto selection |
| Platform | IMC Prosperity 4 |

---

## Status

Competition finished. Repository archived as a portfolio reference for quantitative research workflows, event-driven trading simulation, and systematic parameter optimisation.

---

*Built during IMC Prosperity 4 · May 2025 · Diego Ringleb*
