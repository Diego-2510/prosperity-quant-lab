from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prosperity_quant_lab.exchange import (
    ExchangeError,
    OrderBook,
)


class DataError(ValueError):
    """Raised when historical input cannot be interpreted safely."""


BASE_COLUMNS = {
    "day",
    "timestamp",
    "product",
}


@dataclass(frozen=True)
class Snapshot:
    day: int
    timestamp: int
    books: dict[
        str,
        OrderBook,
    ]
    mids: dict[
        str,
        float,
    ]


def _read_csv(
    path: Path,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            sep=";",
        )

        if len(frame.columns) == 1:
            frame = pd.read_csv(path)

    except (
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise DataError(f"failed to read {path}: {exc}") from exc

    return frame


def _level_numbers(
    columns: Sequence[str],
) -> list[int]:
    levels: set[int] = set()

    for column in columns:
        for prefix in (
            "bid_price_",
            "ask_price_",
            "bid_volume_",
            "ask_volume_",
        ):
            if column.startswith(prefix):
                suffix = column[len(prefix) :]

                if suffix.isdigit():
                    levels.add(int(suffix))

    return sorted(levels)


def load_price_csvs(
    paths: Sequence[str | Path],
) -> pd.DataFrame:
    if not paths:
        raise DataError("at least one price CSV is required")

    frames: list[pd.DataFrame] = []

    for raw_path in paths:
        path = Path(raw_path)

        if not path.is_file():
            raise DataError(f"price CSV does not exist: {path}")

        frame = _read_csv(path)

        missing = sorted(BASE_COLUMNS.difference(frame.columns))

        if missing:
            raise DataError(f"{path.name} missing required columns: " + ", ".join(missing))

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined.empty:
        raise DataError("price CSVs contain no rows")

    for column in (
        "day",
        "timestamp",
    ):
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

        if combined[column].isna().any():
            raise DataError(f"column {column} contains non-numeric values")

    combined["product"] = combined["product"].astype(str).str.strip()

    if (combined["product"] == "").any():
        raise DataError("product contains empty values")

    duplicates = combined.duplicated(
        subset=[
            "day",
            "timestamp",
            "product",
        ],
        keep=False,
    )

    if duplicates.any():
        raise DataError("duplicate (day, timestamp, product) rows are not allowed")

    return combined.sort_values(
        [
            "day",
            "timestamp",
            "product",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _parse_level(
    row: pd.Series,
    level: int,
    side: str,
) -> (
    tuple[
        int,
        int,
    ]
    | None
):
    price_column = f"{side}_price_{level}"

    volume_column = f"{side}_volume_{level}"

    if price_column not in row.index or volume_column not in row.index:
        return None

    price_missing = pd.isna(row[price_column])

    volume_missing = pd.isna(row[volume_column])

    if price_missing and volume_missing:
        return None

    if price_missing != volume_missing:
        raise DataError(f"partial {side} level {level}: price and volume must both be present")

    price = float(row[price_column])

    volume = float(row[volume_column])

    if not np.isfinite(price) or not np.isfinite(volume):
        raise DataError(f"non-finite {side} level {level}")

    if price <= 0 or volume == 0 or volume % 1 != 0 or price % 1 != 0:
        raise DataError(f"invalid {side} level {level}: integer price and non-zero volume required")

    normalized_volume = int(abs(volume))

    if side == "ask":
        normalized_volume = -normalized_volume

    return (
        int(price),
        normalized_volume,
    )


def snapshots_from_prices(
    frame: pd.DataFrame,
) -> list[Snapshot]:
    levels = _level_numbers(frame.columns.tolist())

    if not levels:
        raise DataError("no bid/ask price-volume levels found")

    snapshots: list[Snapshot] = []

    for (
        (
            day,
            timestamp,
        ),
        group,
    ) in frame.groupby(
        [
            "day",
            "timestamp",
        ],
        sort=True,
    ):
        books: dict[
            str,
            OrderBook,
        ] = {}

        mids: dict[
            str,
            float,
        ] = {}

        for _, row in group.iterrows():
            product = str(row["product"])

            buy_orders: dict[
                int,
                int,
            ] = {}

            sell_orders: dict[
                int,
                int,
            ] = {}

            for level in levels:
                bid = _parse_level(
                    row,
                    level,
                    "bid",
                )

                ask = _parse_level(
                    row,
                    level,
                    "ask",
                )

                if bid is not None:
                    buy_orders[bid[0]] = bid[1]

                if ask is not None:
                    sell_orders[ask[0]] = ask[1]

            book = OrderBook(
                buy_orders=buy_orders,
                sell_orders=sell_orders,
            )

            try:
                book.validate()

            except ExchangeError as exc:
                raise DataError(
                    f"invalid book for {product} at ({day}, {timestamp}): {exc}"
                ) from exc

            books[product] = book

            raw_mid = row.get(
                "mid_price",
                np.nan,
            )

            if pd.notna(raw_mid):
                mid = float(raw_mid)

                if not np.isfinite(mid) or mid <= 0:
                    raise DataError(f"invalid mid_price for {product} at ({day}, {timestamp})")

                mids[product] = mid

            elif book.mid is not None:
                mids[product] = float(book.mid)

        snapshots.append(
            Snapshot(
                int(day),
                int(timestamp),
                books,
                mids,
            )
        )

    return snapshots
