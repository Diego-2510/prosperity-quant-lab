from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prosperity_quant_lab.data import (
    DataError,
    load_price_csvs,
)
from prosperity_quant_lab.exchange import (
    ExchangeError,
)
from prosperity_quant_lab.reporting import (
    write_report,
)
from prosperity_quant_lab.simulation import (
    replay,
)
from prosperity_quant_lab.strategies import (
    MeanReversionStrategy,
)

DEFAULT_POSITION_LIMIT = 50


def _parse_position_limit(
    value: str,
) -> tuple[
    str,
    int,
]:
    try:
        (
            product,
            raw_limit,
        ) = value.split(
            "=",
            1,
        )

        limit = int(raw_limit)

    except (
        ValueError,
        TypeError,
    ) as exc:
        raise argparse.ArgumentTypeError("expected PRODUCT=LIMIT") from exc

    product = product.strip()

    if not product or limit <= 0:
        raise argparse.ArgumentTypeError("position limit must use PRODUCT=positive_integer")

    return (
        product,
        limit,
    )


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run a conservative Prosperity historical replay.")
    )

    parser.add_argument(
        "--prices",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="output/replay",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--z-entry",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--order-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--position-limit",
        action="append",
        type=_parse_position_limit,
        default=[],
        metavar="PRODUCT=LIMIT",
    )

    parser.add_argument(
        "--default-position-limit",
        type=int,
        default=DEFAULT_POSITION_LIMIT,
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    args = parse_args(argv)

    if args.default_position_limit <= 0:
        print(
            "error: --default-position-limit must be positive",
            file=sys.stderr,
        )

        return 2

    try:
        prices = load_price_csvs(args.prices)

        products = sorted(prices["product"].unique().tolist())

        overrides = dict(args.position_limit)

        unknown = sorted(set(overrides).difference(products))

        if unknown:
            raise ValueError(
                "position-limit override references unknown product(s): " + ", ".join(unknown)
            )

        limits = {
            product: overrides.get(
                product,
                args.default_position_limit,
            )
            for product in products
        }

        strategy = MeanReversionStrategy(
            window=args.window,
            z_entry=args.z_entry,
            order_size=(args.order_size),
        )

        result = replay(
            prices,
            strategy,
            limits,
        )

        write_report(
            result,
            Path(args.output_dir),
            assumptions={
                "fill_model": ("crossing_orders_only"),
                "passive_queue_model": False,
                "bot_flow_model": False,
                "conversions_modelled": False,
                "default_position_limit": (args.default_position_limit),
                "position_limit_overrides": (overrides),
            },
        )

    except (
        DataError,
        ExchangeError,
        ValueError,
        OSError,
    ) as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )

        return 2

    print(f"final_pnl={result.metrics['final_pnl']:.6f}")

    print(f"fills={result.metrics['fill_count']}")

    print(f"output_dir={args.output_dir}")

    return 0
