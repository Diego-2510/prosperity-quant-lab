from pathlib import Path

import pandas as pd

from prosperity_quant_lab.reporting import write_report
from prosperity_quant_lab.simulation import ReplayResult


def test_write_report_emits_csv_json_and_plot(tmp_path: Path) -> None:
    result = ReplayResult(
        fills=pd.DataFrame(
            [{"product": "A", "price": 101, "quantity": 1, "day": 0, "timestamp": 0}]
        ),
        equity_curve=pd.DataFrame(
            [{"day": 0, "timestamp": 0, "equity": 1.0}, {"day": 0, "timestamp": 100, "equity": 2.0}]
        ),
        final_positions={"A": 1},
        final_cash={"A": -101.0},
        metrics={"final_pnl": 2.0, "max_drawdown": 0.0, "turnover": 101.0, "fill_count": 1},
    )

    paths = write_report(result, tmp_path, assumptions={"fill_model": "test"})

    assert len(paths) == 4
    assert all(path.exists() for path in paths)
    assert '"fill_model": "test"' in (tmp_path / "summary.json").read_text(encoding="utf-8")
