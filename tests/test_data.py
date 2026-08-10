from pathlib import Path

import pandas as pd
import pytest

from prosperity_quant_lab.data import DataError, load_price_csvs, snapshots_from_prices


def test_load_and_parse_prosperity_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        [
            {
                "day": 0,
                "timestamp": 0,
                "product": "A",
                "bid_price_1": 99,
                "bid_volume_1": 5,
                "ask_price_1": 101,
                "ask_volume_1": 6,
                "mid_price": 100,
            }
        ]
    ).to_csv(path, sep=";", index=False)

    frame = load_price_csvs([path])
    snapshots = snapshots_from_prices(frame)

    assert len(snapshots) == 1
    assert snapshots[0].books["A"].buy_orders == {99: 5}
    assert snapshots[0].books["A"].sell_orders == {101: -6}
    assert snapshots[0].mids["A"] == 100.0


def test_duplicate_product_at_same_tick_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    row = {"day": 0, "timestamp": 0, "product": "A"}
    pd.DataFrame([row, row]).to_csv(path, sep=";", index=False)

    with pytest.raises(DataError, match="duplicate"):
        load_price_csvs([path])


def test_partial_book_level_is_rejected() -> None:
    frame = pd.DataFrame(
        [
            {
                "day": 0,
                "timestamp": 0,
                "product": "A",
                "bid_price_1": 99,
                "bid_volume_1": None,
                "ask_price_1": 101,
                "ask_volume_1": 5,
            }
        ]
    )

    with pytest.raises(DataError, match="partial bid level"):
        snapshots_from_prices(frame)


def test_locked_book_is_rejected() -> None:
    frame = pd.DataFrame(
        [
            {
                "day": 0,
                "timestamp": 0,
                "product": "A",
                "bid_price_1": 100,
                "bid_volume_1": 5,
                "ask_price_1": 100,
                "ask_volume_1": 5,
            }
        ]
    )

    with pytest.raises(DataError, match="crossed or locked"):
        snapshots_from_prices(frame)
