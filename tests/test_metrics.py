import pandas as pd

from prosperity_quant_lab.metrics import compute_metrics, max_drawdown


def test_max_drawdown_is_peak_to_trough_currency_drawdown() -> None:
    equity = pd.Series([0.0, 5.0, 3.0, 8.0, 2.0])
    assert max_drawdown(equity) == -6.0


def test_metrics_include_turnover() -> None:
    equity = pd.DataFrame({"equity": [0.0, 2.0]})
    fills = pd.DataFrame({"price": [100, 105], "quantity": [2, -1]})
    metrics = compute_metrics(equity, fills)

    assert metrics["final_pnl"] == 2.0
    assert metrics["turnover"] == 305.0
    assert metrics["fill_count"] == 2
