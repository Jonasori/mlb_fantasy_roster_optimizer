"""
Offline tests for the backtest harness.
Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import datetime

import numpy as np
import pandas as pd

from optimizer.backtest import SEASON_END, assemble_backtest_frame


def _projection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2, 3],
            "Name": ["Kept", "Also Kept", "No Evidence"],
            "player_type": ["hitter", "hitter", "hitter"],
            "PA": [200.0, 150.0, 100.0],
            "R": [25.0, 18.0, 12.0], "HR": [8.0, 5.0, 3.0],
            "RBI": [26.0, 19.0, 13.0], "SB": [4.0, 2.0, 1.0],
            "OPS": [0.780, 0.700, 0.650],
        }
    )


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2], "name": ["Kept", "Also Kept"],
            "group": ["hitting", "hitting"],
            "PA": [300.0, 280.0], "AB": [270.0, 250.0], "H": [70.0, 60.0],
            "HR": [12.0, 7.0], "R": [40.0, 33.0], "RBI": [41.0, 30.0],
            "SB": [6.0, 3.0], "CS": [2.0, 1.0], "BB": [27.0, 25.0],
            "SO": [70.0, 62.0], "SF": [3.0, 3.0],
            "avg": [0.259, 0.240], "obp": [0.330, 0.318],
            "slg": [0.440, 0.390], "babip": [0.300, 0.285],
        }
    )


def _actual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2], "name": ["Kept", "Also Kept"],
            "group": ["hitting", "hitting"],
            "PA": [190.0, 140.0], "AB": [170.0, 125.0], "H": [45.0, 30.0],
            "HR": [9.0, 4.0], "R": [27.0, 16.0], "RBI": [28.0, 17.0],
            "SB": [5.0, 1.0], "CS": [1.0, 1.0], "BB": [18.0, 13.0],
            "SO": [44.0, 33.0], "SF": [2.0, 2.0],
            "avg": [0.265, 0.240], "obp": [0.335, 0.312],
            "slg": [0.455, 0.376], "babip": [0.305, 0.280],
        }
    )


def test_season_end_dates_known():
    assert SEASON_END[2026] == datetime.date(2026, 9, 27), (
        f"2026 season end is {SEASON_END.get(2026)}, expected 2026-09-27"
    )


def test_assemble_joins_and_prefixes():
    frame = assemble_backtest_frame(
        2026, datetime.date(2026, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
    )
    assert len(frame) == 2, (
        f"Expected 2 rows (inner join on players with both evidence and "
        f"outcome), got {len(frame)}"
    )
    assert 3 not in set(frame["MLBAMID"]), (
        "Player 3 has a projection but no evidence and no outcome; he must "
        "not appear — scoring him would credit the model for a phantom."
    )
    for col in ("proj_PA", "proj_OPS", "evid_K_pct", "n_evid", "actual_PA"):
        assert col in frame.columns, f"Missing column {col}: {list(frame.columns)}"

    kept = frame[frame["MLBAMID"] == 1].iloc[0]
    assert kept["proj_PA"] == 200.0, f"proj_PA is {kept['proj_PA']}, expected 200"
    assert kept["actual_PA"] == 190.0, f"actual_PA is {kept['actual_PA']}, expected 190"
    assert kept["n_evid"] == 300.0, f"n_evid is {kept['n_evid']}, expected 300 PA"
    assert abs(kept["evid_K_pct"] - 70.0 / 300.0) < 1e-9, (
        f"evid_K_pct is {kept['evid_K_pct']}, expected {70 / 300}"
    )


def test_actual_ops_is_derived_not_copied():
    frame = assemble_backtest_frame(
        2026, datetime.date(2026, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
    )
    kept = frame[frame["MLBAMID"] == 1].iloc[0]
    assert abs(kept["actual_OPS"] - (0.335 + 0.455)) < 1e-9, (
        f"actual_OPS is {kept['actual_OPS']}, expected obp+slg = 0.790"
    )


def test_rejects_split_outside_season():
    try:
        assemble_backtest_frame(
            2026, datetime.date(2026, 12, 1), _projection(),
            evidence=_evidence(), actual=_actual(),
        )
    except AssertionError as exc:
        assert "split" in str(exc).lower(), (
            f"Expected an assertion about the split date, got: {exc}"
        )
    else:
        raise AssertionError(
            "A split date after the season end must fail loudly — it silently "
            "produces an empty outcome window otherwise."
        )
