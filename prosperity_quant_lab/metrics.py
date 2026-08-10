from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(
    equity: pd.Series,
) -> float:
    if equity.empty:
        return 0.0

    values = equity.astype(float)

    running_peak = values.cummax()

    drawdown = values - running_peak

    return float(drawdown.min())


def compute_metrics(
    equity_curve: pd.DataFrame,
    fills: pd.DataFrame,
) -> dict[
    str,
    float | int,
]:
    if equity_curve.empty:
        return {
            "final_pnl": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "fill_count": 0,
        }

    turnover = 0.0

    if not fills.empty:
        turnover = float(np.abs(fills["price"] * fills["quantity"]).sum())

    return {
        "final_pnl": float(equity_curve["equity"].iloc[-1]),
        "max_drawdown": (max_drawdown(equity_curve["equity"])),
        "turnover": turnover,
        "fill_count": len(fills),
    }
