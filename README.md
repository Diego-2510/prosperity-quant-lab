# Prosperity Quant Lab

> **IMC Prosperity 4 — Competition complete.**  
> **#326 globally · #200 Algo · #10 in the country · 30,703 participants**

A full-cycle quantitative research and trading systems project built for the [IMC Prosperity 4](https://imc-prosperity.notion.site/prosperity-4-wiki) algorithmic trading competition. This repository covers everything from initial market hypothesis to live submission: strategy design, rigorous backtesting, Monte-Carlo simulation, Black-Scholes derivatives analytics, and parameter optimisation under real competition constraints.

---

## Results

| Metric | Result |
|---|---|
| Global rank | **#326** |
| Algo rank | **#200** |
| Country rank | **#10 (France)** |
| Total participants | **30,703** |

---

## What was built

The project was intentionally architected as a **research platform**, not a one-off script. The central deliverable is `round3_backtest_montecarlo.py` (~2,000 lines), a self-contained backtesting and parameter-search pipeline that reproduces Prosperity's exchange mechanics and evaluates strategies systematically.

### Core capabilities

- **Prosperity-compatible fill engine** with dual EXACT (real L1/L2 order books) and APPROX (synthetic mid book) modes — sign conventions, position limits, and `traderData` persistence match the live competition environment exactly
- **Monte-Carlo path generator** supporting block bootstrap and Ornstein-Uhlenbeck mean-reversion models, with per-product method overrides
- **Two-stage coarse-to-fine grid search** with Pareto-optimal parameter selection, robust top-K re-evaluation, and neighbourhood-stability re-ranking to avoid selection bias
- **Black-Scholes + volatility smile diagnostics** for Round 3 voucher products: Newton-Raphson IV extraction, quadratic moneyness smile fit, and three diagnostic plots (IV scatter, IV deviation time series, price deviation time series)
- **Parallel execution** via `joblib` process pool (`--n-workers -1` = all cores), achieving ~10–14× speedup for grid search on a 16-core machine
- **Rich artifact output**: equity curves, MC fan charts, Pareto overlays, heatmaps, sensitivity plots, stability boxplots, smile fit JSON for direct reuse in live traders

### Strategies developed

- Z-score mean-reversion market maker (primary strategy, `mean_reversion.py`)
- Options mean-reversion on fair-value spread using fitted smile IV (`mean_reversion_options.py`)
- Round-specific traders tuned per product group across Rounds 1–4

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Numerical core | NumPy, Pandas |
| Visualisation | Matplotlib |
| Parallelism | joblib, tqdm |
| Derivatives pricing | Custom Black-Scholes engine (European calls, r=0) |
| Optimisation | Grid search + coarse-to-fine Pareto selection |
| Competition platform | IMC Prosperity 4 |

---

## Repository structure

```
prosperity-quant-lab/
├── round3_backtest_montecarlo.py   # Full pipeline: backtest · MC · grid search · BS analytics
├── mean_reversion.py               # Primary competition strategy (z-score market maker)
├── requirements.txt                # Pinned dependencies
└── README.md
```

The pipeline is intentionally **single-file and self-contained**: drop in any `Trader` class, point it at competition CSVs, and get a full evaluation report including parameter recommendations and plots.

---

## Engineering highlights

**Speed without sacrificing correctness.** The grid search runs thousands of strategy configurations on Monte-Carlo paths in minutes by separating a cheap noisy coarse stage from a high-fidelity fine stage. The final winner is validated with a separate top-K re-evaluation pass to remove lucky-draw bias from `nlargest()`.

**Modular strategy interface.** Any `Trader` class exposing `PARAM_SPEC` is automatically grid-searched with no manual wiring. Types are auto-cast; grids can be overridden at CLI level with `--param-grid` without touching the strategy file — enabling rapid iteration during a live competition where time is the binding constraint.

**Derivatives analytics as first-class output.** Round 3 introduced options products. Rather than ad-hoc pricing, the pipeline extracts a full implied-volatility surface, fits a quadratic moneyness smile, and writes the parabola coefficients to `vol_smile_fit.json` — directly reusable by the live trader for fair-value estimation.

---

## Key lessons

1. **Infrastructure compounds.** The time invested in a robust backtesting loop in Round 1 paid back across every subsequent round as a fast, trusted feedback mechanism.
2. **Robustness beats peak performance.** Selecting parameters by Pareto frontier on `(mean, variance)` rather than raw PnL consistently outperformed greedy top-1 selection in out-of-sample evaluation.
3. **Derivatives require a pricing model, not a heuristic.** Moving from ad-hoc voucher pricing to a proper IV surface fit produced measurably better edge estimation for the Round 3 options market.

---

## Status

**Competition finished. Repository archived.**  
Maintained as a portfolio project and reference implementation for quantitative research workflows, event-driven trading simulation, and systematic parameter optimisation.

---

*Built during IMC Prosperity 4 · May 2025 · Diego Ringleb*
