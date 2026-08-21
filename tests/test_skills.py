"""
Offline tests for skill-rate decomposition.
Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import numpy as np
import pandas as pd

from data_prep.skills import add_skill_rates

def _hitting_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2],
            "name": ["Full Season", "Zero PA"],
            "group": ["hitting", "hitting"],
            "PA": [400.0, 0.0], "AB": [360.0, 0.0], "H": [90.0, 0.0],
            "HR": [20.0, 0.0], "R": [50.0, 0.0], "RBI": [60.0, 0.0],
            "SB": [8.0, 0.0], "CS": [2.0, 0.0], "BB": [36.0, 0.0],
            "SO": [100.0, 0.0], "SF": [4.0, 0.0],
            "avg": [0.250, np.nan], "obp": [0.320, np.nan],
            "slg": [0.450, np.nan], "babip": [0.280, np.nan],
        }
    )


def _pitching_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [3],
            "name": ["Starter"],
            "group": ["pitching"],
            "BF": [500.0], "outs": [300.0], "IP": [100.0], "W": [8.0],
            "SV": [0.0], "ER": [40.0], "HA": [90.0], "BBA": [30.0],
            "SOA": [125.0], "HRA": [12.0],
            "groundOuts": [120.0], "airOuts": [80.0],
            "avg": [np.nan], "obp": [np.nan], "slg": [np.nan],
            "babip": [0.290],
        }
    )


def test_hitting_skill_rates():
    out = add_skill_rates(_hitting_frame())
    row = out[out["MLBAMID"] == 1].iloc[0]
    assert abs(row["K_pct"] - 100.0 / 400.0) < 1e-9, (
        f"K_pct is {row['K_pct']}, expected {100 / 400}"
    )
    assert abs(row["BB_pct"] - 36.0 / 400.0) < 1e-9, (
        f"BB_pct is {row['BB_pct']}, expected {36 / 400}"
    )
    assert abs(row["ISO"] - (0.450 - 0.250)) < 1e-9, (
        f"ISO is {row['ISO']}, expected slg - avg = 0.200"
    )
    assert abs(row["SBA_rate"] - 10.0 / 400.0) < 1e-9, (
        f"SBA_rate is {row['SBA_rate']}, expected (SB+CS)/PA = {10 / 400}"
    )
    assert row["n_PA"] == 400.0, f"n_PA is {row['n_PA']}, expected 400"


def test_zero_volume_gives_nan_not_zero():
    out = add_skill_rates(_hitting_frame())
    row = out[out["MLBAMID"] == 2].iloc[0]
    for col in ("K_pct", "BB_pct", "SBA_rate"):
        assert np.isnan(row[col]), (
            f"{col} is {row[col]} for a 0-PA player; must be NaN. A zero rate "
            f"reads as 'elite contact' to any downstream shrinkage."
        )
    assert row["n_PA"] == 0.0, f"n_PA is {row['n_PA']}, expected 0"


def test_pitching_skill_rates():
    out = add_skill_rates(_pitching_frame())
    row = out.iloc[0]
    assert abs(row["K_pct"] - 125.0 / 500.0) < 1e-9, (
        f"K_pct is {row['K_pct']}, expected {125 / 500}"
    )
    assert abs(row["BB_pct"] - 30.0 / 500.0) < 1e-9, (
        f"BB_pct is {row['BB_pct']}, expected {30 / 500}"
    )
    assert abs(row["GB_pct"] - 120.0 / 200.0) < 1e-9, (
        f"GB_pct is {row['GB_pct']}, expected groundOuts/(groundOuts+airOuts)"
    )
    assert abs(row["HRFB"] - 12.0 / 80.0) < 1e-9, (
        f"HRFB is {row['HRFB']}, expected HRA/airOuts"
    )
    assert row["n_BF"] == 500.0, f"n_BF is {row['n_BF']}, expected 500"


def test_does_not_mutate_input():
    frame = _hitting_frame()
    before = list(frame.columns)
    add_skill_rates(frame)
    assert list(frame.columns) == before, (
        "add_skill_rates mutated its input; it must copy first (AGENTS.md)."
    )
