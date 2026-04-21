# IMC Prosperity 4 – Backtesting Engine

Round-agnostischer Python-Backtester für IMC Prosperity 4 mit:

- Prosperity-kompatible Fill-Engine (EXACT & APPROX, korrekte Sign-Konventionen,
  aggregierter Positionslimit-Check pro Produktseite).
- Auto-Discovery offizieller Prosperity-CSVs (`prices_round_<r>_day_<d>*.csv`,
  `trades_round_<r>_day_<d>*.csv`) mit `;`-Delimiter.
- Lokale Datamodel-Minimalversion (`Order`, `OrderDepth`, `TradingState`,
  `Trade`, `Observation`) – nutzt offizielles `datamodel`, wenn verfügbar.
- Default-MarketMaker-Trader + Wrapper für externe `Trader`-Klassen.
- Monte-Carlo-Pfadgenerator (Block- + Residual-Bootstrap mit Jump-Perturbation,
  Cross-Asset-Korrelation bleibt über gemeinsame Zeit-Indizes erhalten).
- Parameter-Registry + Coarse-to-Fine Grid Search.
- `objective_score = mean_profit − risk_lambda · variance`, Pareto-Frontier,
  robuste Top-2-Auswahl über Nachbarschafts-Stabilität.
- Visualisierungen: Equity-Curve, MC-Fan-Chart, Final-PnL-Verteilung,
  Profit-vs-Variance-Scatter, Pareto-Plot, Heatmap der Top-2-Parameter,
  Sensitivität `mathematical_low`, Top-Ranking, Stabilitäts-Boxplot.
- Artefakte als CSV + PNG.

## Usage

```bash
python round3_backtest_montecarlo.py \
    --data-dir ./data \
    --out-dir  ./bt_out \
    [--trader path/to/trader.py] \
    [--round 3] \
    [--n-paths 1000] \
    [--seed 42] \
    [--risk-lambda 0.5]
```

Funktioniert mit beliebigen Round-/Day-Kombinationen.
Fehlen Produktlimits, werden sie konservativ aus beobachtetem Volumen
abgeleitet (mit `ASSUMPTION`-Markierung). Keine erfundenen P4-Regeln.

## Outputs

- `full_grid_results.csv`
- `top_parameter_pairs.csv`
- `pareto_front.csv`
- `monte_carlo_summary.csv`
- `feature_diagnostics.csv`
- `plot_*.png`
