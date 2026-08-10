import os
import subprocess
import sys
from pathlib import Path


def test_cli_runs_on_example_dataset(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosperity_quant_lab",
            "--prices",
            str(repo / "examples" / "mini_prices.csv"),
            "--window",
            "3",
            "--z-entry",
            "1.0",
            "--order-size",
            "1",
            "--position-limit",
            "ALPHA=10",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=repo,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "equity_curve.csv").exists()


def test_cli_returns_nonzero_for_missing_input(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "prosperity_quant_lab", "--prices", str(tmp_path / "missing.csv")],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "error:" in result.stderr


def test_main_directly_writes_report(tmp_path: Path) -> None:
    from prosperity_quant_lab.cli import main

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "direct"
    code = main(
        [
            "--prices",
            str(repo / "examples" / "mini_prices.csv"),
            "--window",
            "3",
            "--z-entry",
            "1.0",
            "--order-size",
            "1",
            "--position-limit",
            "ALPHA=10",
            "--output-dir",
            str(out),
        ]
    )

    assert code == 0
    assert (out / "fills.csv").exists()
    assert (out / "equity_curve.png").exists()


def test_main_directly_returns_nonzero_for_missing_input(tmp_path: Path) -> None:
    from prosperity_quant_lab.cli import main

    assert main(["--prices", str(tmp_path / "missing.csv")]) == 2


def test_main_rejects_non_positive_default_position_limit(tmp_path: Path) -> None:
    from prosperity_quant_lab.cli import main

    repo = Path(__file__).resolve().parents[1]
    assert (
        main(
            [
                "--prices",
                str(repo / "examples" / "mini_prices.csv"),
                "--default-position-limit",
                "0",
            ]
        )
        == 2
    )
