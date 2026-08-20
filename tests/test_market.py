"""
Offline tests for data_prep fetch helpers (market parsing, identity ages).
No network access. Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import pandas as pd

from data_prep.mlb_api import age_from_birth_date
from data_prep.raw_io import RAW_DIR, raw_path

from data_prep.market import MARKET_FETCHERS, flip_last_first, parse_dollar_series


def test_parse_dollar_series():
    parsed = parse_dollar_series(pd.Series(["$78.64", "$107", "$1,234.50", "", None]))
    assert parsed.tolist()[:3] == [78.64, 107.0, 1234.50], (
        f"Dollar strings parsed wrong: {parsed.tolist()}"
    )
    assert parsed.iloc[3:].isna().all(), (
        f"Blank/None salaries should be NaN, got {parsed.iloc[3:].tolist()}"
    )
    assert parsed.dtype == float, f"Expected float dtype, got {parsed.dtype}"


def test_flip_last_first():
    cases = {
        "Ohtani, Shohei": "Shohei Ohtani",
        "Ohtani-H, Shohei": "Shohei Ohtani-H",
        "Witt Jr., Bobby": "Bobby Witt Jr.",
        "Shohei Ohtani": "Shohei Ohtani",
    }
    for raw, expected in cases.items():
        assert flip_last_first(raw) == expected, (
            f"flip_last_first({raw!r}) gave {flip_last_first(raw)!r}, "
            f"expected {expected!r}"
        )


def test_market_fetchers_map_to_raw_paths():
    assert set(MARKET_FETCHERS) == {"ottoneu", "adp", "espn", "hkb"}, (
        f"Unexpected market sources: {sorted(MARKET_FETCHERS)}. The raw layer "
        f"expects data/raw/market/{{ottoneu,adp,espn,hkb}}/."
    )
    for name in MARKET_FETCHERS:
        path = raw_path(f"market/{name}")
        assert path.parent == RAW_DIR / "market" / name, (
            f"Source '{name}' would write to {path}, not under "
            f"{RAW_DIR / 'market' / name}"
        )
        assert path.name.count("-") == 2 and path.suffix == ".parquet", (
            f"Snapshot filename should be YYYY-MM-DD.parquet, got {path.name}"
        )


def test_age_from_birth_date_birthday_boundary():
    # Reference 2026-08-20: birthday tomorrow is still 29, today or earlier is 30.
    ages = age_from_birth_date(
        pd.Series(["1996-08-21", "1996-08-20", "1996-08-19", None]),
        on=pd.Timestamp("2026-08-20"),
    )
    assert ages.tolist()[:3] == [29.0, 30.0, 30.0], (
        f"Birthday boundary handled wrong: {ages.tolist()}"
    )
    assert pd.isna(ages.iloc[3]), (
        f"Missing birth_date should give NaN, got {ages.iloc[3]}"
    )


def test_salary_momentum_sign():
    last_10 = parse_dollar_series(pd.Series(["$81.60", "$58.60"]))
    median = parse_dollar_series(pd.Series(["$78.00", "$69.00"]))
    momentum = (last_10 - median).round(2).tolist()
    assert momentum == [3.60, -10.40], (
        f"salary_momentum should be last_10 - median, got {momentum}"
    )
