from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from prosperity_quant_lab.simulation import (
    ReplayResult,
)


def write_report(
    result: ReplayResult,
    output_dir: str | Path,
    *,
    assumptions: Mapping[
        str,
        object,
    ]
    | None = None,
) -> list[Path]:
    out = Path(output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    fills_path = out / "fills.csv"

    equity_path = out / "equity_curve.csv"

    summary_path = out / "summary.json"

    plot_path = out / "equity_curve.png"

    result.fills.to_csv(
        fills_path,
        index=False,
    )

    result.equity_curve.to_csv(
        equity_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            {
                "metrics": dict(result.metrics),
                "final_positions": dict(result.final_positions),
                "final_cash": dict(result.final_cash),
                "assumptions": dict(assumptions or {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            4,
        )
    )

    ax.plot(result.equity_curve["equity"].to_numpy())

    ax.set_xlabel("Replay tick")

    ax.set_ylabel("Marked P&L")

    ax.set_title("Historical replay equity curve")

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return [
        fills_path,
        equity_path,
        summary_path,
        plot_path,
    ]
