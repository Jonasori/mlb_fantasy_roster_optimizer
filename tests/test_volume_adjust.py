"""
Offline tests for the RoS volume correction.
Per AGENTS.md: no classes, no fixtures, no mocking.

These tests guard invariants that NOTHING else in the codebase enforces:
nothing checks OPS against PA, and scaling PA does not scale R/HR/RBI/SB.
"""

import numpy as np
import pandas as pd

from data_prep.volume_adjust import adjust_projection_volume

_COEFFS = {
    "b0": -0.10, "b_age": -0.01, "b_talent": 0.50, "b_slump": 0.40,
    "min_factor": 0.25, "max_factor": 2.0,
}


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["Hitter A", "Hitter B", "Pitcher C"],
            "MLBAMID": [1, 2, 3],
            "player_type": ["hitter", "hitter", "pitcher"],
            "age": [29.0, 34.0, 27.0],
            "PA": [130.0, 120.0, 0.0], "AB": [117.0, 108.0, 0.0],
            "R": [16.0, 14.0, 0.0], "HR": [5.0, 4.0, 0.0],
            "RBI": [17.0, 15.0, 0.0], "SB": [4.0, 1.0, 0.0],
            "OPS": [0.780, 0.700, 0.0],
            "IP": [0.0, 0.0, 40.0], "W": [0.0, 0.0, 3.0],
            "SV": [0.0, 0.0, 0.0], "K": [0.0, 0.0, 42.0],
            "ERA": [0.0, 0.0, 3.50], "WHIP": [0.0, 0.0, 1.10],
        }
    )


def _ytd() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2, 3],
            "group": ["hitting", "hitting", "pitching"],
            "n_PA": [400.0, 380.0, np.nan],
            "n_BF": [np.nan, np.nan, 500.0],
            "ytd_OPS": [0.800, 0.560, np.nan],
        }
    )


def test_counting_stats_scale_with_volume():
    """The invariant nothing else enforces: scaling PA must scale R/HR/RBI/SB."""
    before = _players()
    after = adjust_projection_volume(before, _ytd(), _COEFFS)
    a = after[after["Name"] == "Hitter A"].iloc[0]
    b = before[before["Name"] == "Hitter A"].iloc[0]
    factor = a["PA"] / b["PA"]
    for stat in ("AB", "R", "HR", "RBI", "SB"):
        assert abs(a[stat] - b[stat] * factor) < 1e-9, (
            f"{stat} is {a[stat]} but PA scaled by {factor:.4f} from {b[stat]}, "
            f"so it should be {b[stat] * factor}. A player who plays more games "
            f"and scores the same runs is a silent corruption."
        )
    assert abs(a["PA_adj_factor"] - factor) < 1e-9, (
        f"PA_adj_factor {a['PA_adj_factor']} disagrees with the applied "
        f"factor {factor}"
    )


def test_rates_are_untouched():
    after = adjust_projection_volume(_players(), _ytd(), _COEFFS)
    before = _players()
    for name, col in (("Hitter A", "OPS"), ("Pitcher C", "ERA"), ("Pitcher C", "WHIP")):
        a = after[after["Name"] == name].iloc[0][col]
        b = before[before["Name"] == name].iloc[0][col]
        assert a == b, (
            f"{name}'s {col} changed from {b} to {a}. This is the VOLUME "
            f"corrector; rate correction is Part 2b and out of scope."
        )


def test_opposite_type_columns_stay_exactly_zero():
    after = adjust_projection_volume(_players(), _ytd(), _COEFFS)
    hitters = after[after["player_type"] == "hitter"]
    pitchers = after[after["player_type"] == "pitcher"]
    for col in ("IP", "W", "SV", "K", "ERA", "WHIP"):
        assert (hitters[col] == 0.0).all(), (
            f"Hitter {col} is not exactly 0.0: {hitters[col].tolist()}. "
            f"build.py:236 zeroes these and MEW gains a phantom term otherwise."
        )
    for col in ("PA", "R", "HR", "RBI", "SB", "OPS"):
        assert (pitchers[col] == 0.0).all(), (
            f"Pitcher {col} is not exactly 0.0: {pitchers[col].tolist()}"
        )
    # A mask bug (e.g. is_hitter/is_pitcher swapped when building the adj
    # factors) would still pass the checks above, since 0 times anything is
    # 0 — the fixture's opposite-type stats start at zero, so multiplying
    # them by a nontrivial factor is invisible there. The factor columns
    # themselves are not fixed points of multiplication, so pin them too:
    # a mis-scoped mask would give hitters a non-1.0 IP_adj_factor even
    # though the IP column it multiplies happens to already be zero.
    assert (hitters["IP_adj_factor"] == 1.0).all(), (
        f"Hitter IP_adj_factor is not exactly 1.0: "
        f"{hitters['IP_adj_factor'].tolist()}. The volume model must not "
        f"apply to a group it doesn't cover, even where the effect on IP "
        f"(0 * factor = 0) would otherwise be invisible."
    )
    assert (pitchers["PA_adj_factor"] == 1.0).all(), (
        f"Pitcher PA_adj_factor is not exactly 1.0: "
        f"{pitchers['PA_adj_factor'].tolist()}"
    )


def test_never_scales_a_player_to_zero_volume():
    """Zero volume drops a player from FV's z-population and benches IL players."""
    extreme = {**_COEFFS, "b_slump": 50.0}
    after = adjust_projection_volume(_players(), _ytd(), extreme)
    hitters = after[after["player_type"] == "hitter"]
    assert (hitters["PA"] > 0).all(), (
        f"A hitter reached zero PA: {hitters[['Name', 'PA']].to_dict('records')}. "
        f"Clamp to min_factor — zero volume silently removes him from FV's "
        f"ratio z-population and permanently benches him if he is on the IL."
    )
    assert (hitters["PA_adj_factor"] >= _COEFFS["min_factor"] - 1e-9).all(), (
        "min_factor clamp was not applied"
    )


def test_slumping_player_loses_playing_time():
    """MGL: cold hitters lose ~30 PA that projections do not anticipate."""
    after = adjust_projection_volume(_players(), _ytd(), _COEFFS)
    hot = after[after["Name"] == "Hitter A"].iloc[0]["PA_adj_factor"]
    cold = after[after["Name"] == "Hitter B"].iloc[0]["PA_adj_factor"]
    assert cold < hot, (
        f"Hitter B is 140 points of OPS below his projection and Hitter A is "
        f"20 above, yet B's factor ({cold:.3f}) is not below A's ({hot:.3f})."
    )


def test_players_with_no_ytd_are_left_alone():
    players = _players()
    after = adjust_projection_volume(players, _ytd().iloc[:0], _COEFFS)
    hitters = after[after["player_type"] == "hitter"]
    assert (hitters["PA_adj_factor"] == 1.0).all(), (
        f"A player with no YTD evidence must pass through unchanged, got "
        f"{hitters['PA_adj_factor'].tolist()}"
    )


def test_does_not_mutate_input():
    players = _players()
    before_pa = players["PA"].tolist()
    adjust_projection_volume(players, _ytd(), _COEFFS)
    assert players["PA"].tolist() == before_pa, (
        "adjust_projection_volume mutated its input; it must copy first."
    )
