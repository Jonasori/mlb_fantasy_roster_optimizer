"""
Offline tests for the StatsAPI byDateRange parser.
No network access. Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import numpy as np
import pandas as pd

from data_prep.statsapi_stats import parse_rate, parse_stat_splits

_HITTING_PAYLOAD = {
    "stats": [
        {
            "splits": [
                {
                    "player": {"id": 663728, "fullName": "Cal Raleigh"},
                    "numTeams": 1,
                    "stat": {
                        "plateAppearances": 392, "atBats": 340, "hits": 53,
                        "homeRuns": 17, "runs": 42, "rbi": 51, "stolenBases": 1,
                        "caughtStealing": 0, "baseOnBalls": 45, "strikeOuts": 125,
                        "sacFlies": 4, "avg": ".156", "obp": ".273", "slg": ".296",
                        "babip": ".196",
                    },
                },
                {
                    "player": {"id": 1, "fullName": "No Rate Guy"},
                    "numTeams": 2,
                    "stat": {
                        "plateAppearances": 0, "atBats": 0, "hits": 0,
                        "homeRuns": 0, "runs": 0, "rbi": 0, "stolenBases": 0,
                        "caughtStealing": 0, "baseOnBalls": 0, "strikeOuts": 0,
                        "sacFlies": 0, "avg": ".---", "obp": ".---", "slg": ".---",
                        "babip": ".---",
                    },
                },
            ]
        }
    ]
}

_PITCHING_PAYLOAD = {
    "stats": [
        {
            "splits": [
                {
                    "player": {"id": 694973, "fullName": "Paul Skenes"},
                    "numTeams": 1,
                    "stat": {
                        "battersFaced": 500, "outs": 229, "inningsPitched": "76.1",
                        "wins": 8, "saves": 0, "earnedRuns": 20, "hits": 55,
                        "baseOnBalls": 20, "strikeOuts": 95, "homeRuns": 6,
                        "groundOuts": 70, "airOuts": 60, "babip": ".280",
                    },
                }
            ]
        }
    ]
}


def test_parse_rate_handles_string_and_sentinel():
    parsed = parse_rate(pd.Series([".319", ".---", None, ".000"]))
    assert parsed.iloc[0] == 0.319, f"'.319' parsed as {parsed.iloc[0]}, expected 0.319"
    assert np.isnan(parsed.iloc[1]), f"'.---' should be NaN, got {parsed.iloc[1]}"
    assert np.isnan(parsed.iloc[2]), f"None should be NaN, got {parsed.iloc[2]}"
    assert parsed.iloc[3] == 0.0, f"'.000' parsed as {parsed.iloc[3]}, expected 0.0"
    assert parsed.dtype == float, f"Expected float dtype, got {parsed.dtype}"


def test_parse_stat_splits_hitting():
    df = parse_stat_splits(_HITTING_PAYLOAD, "hitting")
    assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
    raleigh = df[df["MLBAMID"] == 663728].iloc[0]
    assert raleigh["PA"] == 392, f"PA parsed as {raleigh['PA']}, expected 392"
    assert raleigh["SO"] == 125, f"SO parsed as {raleigh['SO']}, expected 125"
    assert abs(raleigh["slg"] - 0.296) < 1e-9, (
        f"slg parsed as {raleigh['slg']}, expected 0.296"
    )
    assert raleigh["group"] == "hitting", f"group is {raleigh['group']}"
    assert np.isnan(df[df["MLBAMID"] == 1].iloc[0]["avg"]), (
        "'.---' avg should parse to NaN"
    )


def test_parse_stat_splits_pitching_uses_outs_not_innings_string():
    df = parse_stat_splits(_PITCHING_PAYLOAD, "pitching")
    row = df.iloc[0]
    assert row["outs"] == 229, f"outs parsed as {row['outs']}, expected 229"
    # 229/3 = 76.333..., NOT the 76.1 that a naive float() of "76.1" would give.
    assert abs(row["IP"] - 229 / 3) < 1e-9, (
        f"IP is {row['IP']}, expected {229 / 3}. "
        f"inningsPitched '76.1' means 76 1/3 — derive IP from outs."
    )
    assert abs(row["IP"] - 76.1) > 0.2, (
        f"IP is {row['IP']}, which looks like a naive float('76.1'). Use outs/3."
    )
    assert row["BF"] == 500, f"BF parsed as {row['BF']}, expected 500"


def test_parse_stat_splits_no_duplicate_players():
    df = parse_stat_splits(_HITTING_PAYLOAD, "hitting")
    assert not df["MLBAMID"].duplicated().any(), (
        "byDateRange aggregates traded players across teams; duplicate MLBAMIDs "
        "mean the parser is splitting on team."
    )
