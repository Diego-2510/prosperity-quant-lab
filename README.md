# Prosperity Quant Lab

A tested research toolkit built from the quantitative infrastructure developed during **IMC Prosperity 4**.

**Competition result:** #326 globally, #200 in the algorithmic ranking, #10 in-country, among 30,703 participants.

The current repository is intentionally conservative about what its historical replay can establish. It provides a modular data parser, crossing-order replay kernel, deterministic simulation, grid-search utilities, moving-block bootstrap, options-pricing primitives, metrics, and reproducible reporting.

It does **not** claim to reproduce the Prosperity exchange exactly.

## Why This Repository Was Refactored

The original project grew during a live competition and accumulated:

- a single backtesting/Monte-Carlo script of more than 4,000 lines,
- raw competition CSVs committed directly to the repository,
- generated backtest plots committed as source artifacts,
- round-specific strategy scripts mixed with reusable infrastructure,
- assumptions and verified mechanics in the same implementation,
- public wording such as “replicating Prosperity exchange mechanics exactly” despite unmodelled passive bot flow.

Sprint 0 separates reusable research infrastructure from historical competition artifacts and adds executable tests around the most important exchange assumptions.

## Modules

```text
prosperity_quant_lab/
├── data.py          # Prosperity CSV validation and snapshot construction
├── exchange.py      # Orders, books, position limits and crossing fills
├── simulation.py    # Deterministic historical replay + block bootstrap
├── strategies.py    # Strategy protocol + reproducible example strategy
├── optimization.py  # Deterministic parameter-grid evaluation
├── metrics.py       # P&L, drawdown, turnover and fill metrics
├── pricing.py       # Black-Scholes call pricing and implied volatility
├── reporting.py     # CSV, JSON and equity-curve artifacts
├── cli.py
└── __main__.py
```

Historical competition strategy files are kept separately under:

```text
archive/competition_strategies/
```

They are retained as historical artifacts and are not presented as part of the tested library.

## Replay Model

The replay kernel models **immediately marketable crossing orders** against historical visible order-book snapshots.

Implemented behavior:

- positive order quantity = buy,
- negative order quantity = sell,
- historical sell-book volumes are normalized to negative values,
- marketable buys consume asks from lowest price upward,
- marketable sells consume bids from highest price downward,
- execution occurs at the historical resting price,
- buy and sell submitted quantities are checked independently against the absolute position limit,
- if one submitted side would breach the position limit, that entire side is rejected before matching.

The golden tests lock down these semantics.

## Explicit Non-Goals

The replay does not model:

- passive queue position,
- future bot flow into resting quotes,
- hidden liquidity,
- latency,
- conversion mechanics,
- persistent unfilled orders after the current replay tick,
- exchange priority beyond visible price ordering,
- any mechanism that cannot be reconstructed from the supplied snapshot.

Therefore output is a **historical replay estimate**, not exact competition P&L.

See [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

## Quick Start

```bash
git clone https://github.com/Diego-2510/prosperity-quant-lab.git
cd prosperity-quant-lab

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run the included deterministic mini-example:

```bash
python -m prosperity_quant_lab   --prices examples/mini_prices.csv   --window 3   --z-entry 1.0   --order-size 1   --position-limit ALPHA=10   --output-dir output/demo
```

Expected output artifacts:

```text
output/demo/
├── fills.csv
├── equity_curve.csv
├── equity_curve.png
└── summary.json
```

`summary.json` also records the replay assumptions used for the run.

## Position Limits

Research runs should pass explicit position limits:

```bash
--position-limit PRODUCT_A=50 --position-limit PRODUCT_B=100
```

A fallback of `50` exists only for the demo path when no override is supplied. It is an explicit modelling assumption and not a claim about a specific Prosperity round.

## Example Data

Only a tiny synthetic-style book fixture is committed:

```text
examples/mini_prices.csv
```

Large raw competition exports are deliberately excluded from the repository. This keeps fresh clones small and makes the quickstart reproducible without bundling multi-megabyte competition datasets.

## Testing

```bash
python -m ruff format --check .
python -m ruff check .

python -m compileall -q   prosperity_quant_lab   tests

MPLBACKEND=Agg python -m pytest

python -m pip_audit -r requirements.txt
```

The tests cover:

- Prosperity CSV parsing,
- duplicate snapshot rejection,
- malformed order-book levels,
- locked/crossed historical books,
- multi-level buy fills,
- multi-level sell fills,
- aggregate buy-side position-limit rejection,
- aggregate sell-side position-limit rejection,
- absence of invented passive fills,
- deterministic mark-to-market replay,
- moving-block-bootstrap reproducibility,
- P&L/drawdown/turnover metrics,
- deterministic grid expansion,
- grid-search execution,
- Black-Scholes pricing,
- implied-volatility recovery,
- reporting artifacts,
- CLI success and error handling.

## Golden Exchange Tests

The exchange tests are intentionally narrow.

They prove that the local replay kernel behaves consistently with the documented assumptions used by this project. They do **not** prove that every Prosperity exchange mechanic has been reconstructed.

This distinction is important: a tested approximation is more credible than an untestable “exact” simulator claim.

## Monte Carlo

`moving_block_bootstrap(...)` provides a deterministic, seeded moving-block-bootstrap primitive for return-series experiments.

It preserves local blocks from the observed return sample but does not establish a valid market-generating process or prove future strategy profitability.

## Optimization

`grid_search(...)` evaluates a supplied strategy factory over the Cartesian parameter grid and records replay metrics.

The implementation is deterministic and easy to audit. Grid search does not itself prevent overfitting; train/validation or walk-forward design remains the responsibility of the research experiment.

## Options Pricing

The package includes:

- European Black-Scholes call pricing,
- implied-volatility inversion by bounded bisection,
- arbitrage-bound validation.

These functions are research primitives. They do not imply that the competition products exactly satisfied Black-Scholes assumptions.

## CI

GitHub Actions runs on Python 3.12 and 3.13:

- Ruff formatting,
- Ruff linting,
- bytecode compilation,
- pytest with an 85% package coverage threshold,
- dependency audit.

Dependabot monitors Python and GitHub Actions dependencies.

## Repository Structure

```text
prosperity-quant-lab/
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       └── ci.yml
├── archive/
│   ├── README.md
│   └── competition_strategies/
├── docs/
│   └── ASSUMPTIONS.md
├── examples/
│   └── mini_prices.csv
├── prosperity_quant_lab/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── data.py
│   ├── exchange.py
│   ├── metrics.py
│   ├── optimization.py
│   ├── pricing.py
│   ├── reporting.py
│   ├── simulation.py
│   └── strategies.py
├── tests/
│   ├── test_cli.py
│   ├── test_data.py
│   ├── test_exchange.py
│   ├── test_metrics.py
│   ├── test_optimization.py
│   ├── test_pricing.py
│   ├── test_reporting.py
│   └── test_simulation.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Known Limitations

- The replay only models crossing fills.
- Historical book snapshots do not reveal queue position.
- Passive fills are deliberately not inferred.
- The simulator does not model conversions.
- Transaction costs are not added unless they are explicitly represented by a future experiment.
- Position-limit fallback values are assumptions unless explicitly overridden.
- Grid search can overfit historical data.
- The block bootstrap is not a calibrated market model.
- Black-Scholes assumptions may not match all competition products.
- Historical competition rankings do not imply future trading performance.

## Status

IMC Prosperity 4 is complete.

This repository is maintained as a tested quantitative-research portfolio project rather than as a live-trading system.

## License

MIT. See [`LICENSE`](LICENSE).
