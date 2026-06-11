"""
Command-line entry: build silver table and write CSV or Parquet.

Usage:
    cd data_prep && uv run build-silver-table
    cd data_prep && uv run python -m data_prep.cli --output data/silver_table.csv
"""

import argparse
from pathlib import Path

from .build import build_silver_table
from .config import HITTER_PROJ_PATH, PITCHER_PROJ_PATH, SILVER_TABLE_DEFAULT_PATH
from .silver_io import write_silver_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build silver players table (FanGraphs + Fantrax + MLB ages) and write to disk."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(SILVER_TABLE_DEFAULT_PATH),
        help=f"Output path (.csv or .parquet). Default: {SILVER_TABLE_DEFAULT_PATH}",
    )
    parser.add_argument(
        "--hitter",
        type=str,
        default=None,
        help="Override hitter projections CSV (default: from config)",
    )
    parser.add_argument(
        "--pitcher",
        type=str,
        default=None,
        help="Override pitcher projections CSV (default: from config)",
    )
    parser.add_argument(
        "--skip-mlb-api",
        action="store_true",
        help="Do not fetch ages from MLB Stats API (keeps ages from Fantrax merge only)",
    )
    args = parser.parse_args()

    hitter = args.hitter or HITTER_PROJ_PATH
    pitcher = args.pitcher or PITCHER_PROJ_PATH

    assert Path(hitter).is_file(), f"Hitter projections file not found: {hitter}"
    assert Path(pitcher).is_file(), f"Pitcher projections file not found: {pitcher}"

    players = build_silver_table(
        hitter,
        pitcher,
        skip_mlb_api=args.skip_mlb_api,
    )
    write_silver_table(players, args.output)


if __name__ == "__main__":
    main()
