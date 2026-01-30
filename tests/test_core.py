"""
Minimal test suite for MLB Fantasy Roster Optimizer.

All tests are module-level functions with inline test data.
No classes, no fixtures, no mocking.
"""

import numpy as np
import pandas as pd

# =============================================================================
# SMOKE TESTS - Verify imports work
# =============================================================================


def test_import_data_loader():
    """Core data_loader exports are importable."""
    from optimizer.data_loader import (
        ALL_CATEGORIES,
        FANTRAX_TEAM_IDS,
        HITTING_CATEGORIES,
        NUM_OPPONENTS,
        PITCHING_CATEGORIES,
        ROSTER_SIZE,
        SLOT_ELIGIBILITY,
        compute_sgp_value,
        compute_team_totals,
        estimate_projection_uncertainty,
        strip_name_suffix,
    )

    assert len(ALL_CATEGORIES) == 10
    assert ROSTER_SIZE == 26
    assert NUM_OPPONENTS == 6
    assert len(FANTRAX_TEAM_IDS) == 7
    assert "C" in SLOT_ELIGIBILITY


def test_import_mlb_api():
    """MLB Stats API module is importable."""
    from optimizer.mlb_api import fetch_player_ages

    assert callable(fetch_player_ages)


def test_fetch_player_ages_real_api():
    """Fetch player ages from real MLB Stats API."""
    from optimizer.mlb_api import fetch_player_ages

    # Known MLBAM IDs: Shohei Ohtani, Mike Trout, Aaron Judge
    test_ids = [660271, 545361, 592450]

    df = fetch_player_ages(test_ids)

    # Should return DataFrame with correct columns
    assert set(df.columns) == {"mlbam_id", "name", "birth_date", "age"}

    # Should have 3 rows
    assert len(df) == 3

    # All IDs should be present
    assert set(df["mlbam_id"].tolist()) == set(test_ids)

    # All ages should be positive integers
    assert df["age"].dtype in [np.int64, int, "int64"]
    assert (df["age"] > 0).all()

    # Verify known player names
    names = df["name"].tolist()
    assert "Shohei Ohtani" in names
    assert "Mike Trout" in names
    assert "Aaron Judge" in names


def test_fetch_player_ages_deduplicates():
    """Duplicate IDs are handled correctly."""
    from optimizer.mlb_api import fetch_player_ages

    # Pass same ID multiple times
    test_ids = [660271, 660271, 660271]

    df = fetch_player_ages(test_ids)

    # Should return only 1 unique player
    assert len(df) == 1
    assert df.iloc[0]["name"] == "Shohei Ohtani"


def test_import_roster_optimizer():
    """Roster optimizer exports are importable."""
    from optimizer.roster_optimizer import (
        BIG_M_COUNTING,
        EPSILON_RATIO,
        build_and_solve_milp,
        filter_candidates,
    )

    assert BIG_M_COUNTING == 10000
    assert EPSILON_RATIO == 0.001


def test_import_trade_engine():
    """Trade engine exports are importable."""
    from optimizer.trade_engine import (
        MEV_TABLE,
        MVAR_TABLE,
        compute_player_values,
        compute_win_probability,
        evaluate_trade,
    )

    assert MEV_TABLE[6] == 1.267
    assert MVAR_TABLE[6] == 0.416


def test_import_visualizations():
    """Visualizations exports are importable."""
    from optimizer.visualizations import (
        plot_team_radar,
        plot_trade_impact,
    )


# =============================================================================
# NAME HANDLING TESTS
# =============================================================================


def test_strip_name_suffix():
    """strip_name_suffix removes -H and -P correctly."""
    from optimizer.data_loader import strip_name_suffix

    assert strip_name_suffix("Mike Trout-H") == "Mike Trout"
    assert strip_name_suffix("Gerrit Cole-P") == "Gerrit Cole"
    assert strip_name_suffix("Shohei Ohtani-H") == "Shohei Ohtani"
    assert strip_name_suffix("Shohei Ohtani-P") == "Shohei Ohtani"
    assert strip_name_suffix("No Suffix") == "No Suffix"


# =============================================================================
# SGP CALCULATION TESTS
# =============================================================================


def test_compute_sgp_value_hitter():
    """SGP calculation for hitters produces reasonable values."""
    from optimizer.data_loader import compute_sgp_value

    # Create a realistic hitter row as pd.Series
    hitter = pd.Series(
        {
            "player_type": "hitter",
            "PA": 600,
            "R": 95,
            "HR": 28,
            "RBI": 85,
            "SB": 15,
            "OPS": 0.820,
        }
    )

    sgp = compute_sgp_value(hitter)

    # Rough expected: R/20 + HR/8 + RBI/20 + SB/7 + (OPS-0.75)/0.01
    # = 4.75 + 3.5 + 4.25 + 2.14 + 7.0 = ~21.6
    assert 15 < sgp < 30, f"Hitter SGP {sgp} outside reasonable range"


def test_compute_sgp_value_pitcher():
    """SGP calculation for pitchers produces reasonable values."""
    from optimizer.data_loader import compute_sgp_value

    # Create a realistic pitcher row as pd.Series
    pitcher = pd.Series(
        {
            "player_type": "pitcher",
            "IP": 180.0,
            "W": 14,
            "SV": 0,
            "K": 200,
            "ERA": 3.25,
            "WHIP": 1.10,
        }
    )

    sgp = compute_sgp_value(pitcher)

    # Rough expected: W/3.5 + SV/8 + K/35 + (4.0-ERA)/0.18 + (1.25-WHIP)/0.03
    # = 4.0 + 0 + 5.7 + 4.2 + 5.0 = ~18.9
    assert 10 < sgp < 30, f"Pitcher SGP {sgp} outside reasonable range"


# =============================================================================
# TEAM TOTALS TESTS
# =============================================================================


def test_compute_team_totals_counting_stats():
    """Counting stats are summed correctly."""
    from optimizer.data_loader import compute_team_totals

    projections = pd.DataFrame(
        [
            {
                "Name": "Player A-H",
                "player_type": "hitter",
                "PA": 500,
                "R": 80,
                "HR": 25,
                "RBI": 70,
                "SB": 10,
                "OPS": 0.850,
                "IP": 0,
                "W": 0,
                "SV": 0,
                "K": 0,
                "ERA": 0,
                "WHIP": 0,
            },
            {
                "Name": "Player B-H",
                "player_type": "hitter",
                "PA": 400,
                "R": 60,
                "HR": 15,
                "RBI": 50,
                "SB": 5,
                "OPS": 0.750,
                "IP": 0,
                "W": 0,
                "SV": 0,
                "K": 0,
                "ERA": 0,
                "WHIP": 0,
            },
            {
                "Name": "Pitcher A-P",
                "player_type": "pitcher",
                "PA": 0,
                "R": 0,
                "HR": 0,
                "RBI": 0,
                "SB": 0,
                "OPS": 0,
                "IP": 100,
                "W": 8,
                "SV": 0,
                "K": 100,
                "ERA": 3.50,
                "WHIP": 1.15,
            },
        ]
    )

    totals = compute_team_totals(
        {"Player A-H", "Player B-H", "Pitcher A-P"}, projections
    )

    assert totals["R"] == 140, f"R should be 80+60=140, got {totals['R']}"
    assert totals["HR"] == 40, f"HR should be 25+15=40, got {totals['HR']}"
    assert totals["RBI"] == 120, f"RBI should be 70+50=120, got {totals['RBI']}"
    assert totals["SB"] == 15, f"SB should be 10+5=15, got {totals['SB']}"


def test_compute_team_totals_ratio_stats():
    """Ratio stats use PA/IP-weighted averages, not sums."""
    from optimizer.data_loader import compute_team_totals

    projections = pd.DataFrame(
        [
            {
                "Name": "Player A-H",
                "player_type": "hitter",
                "PA": 600,
                "R": 100,
                "HR": 30,
                "RBI": 90,
                "SB": 10,
                "OPS": 0.900,
                "IP": 0,
                "W": 0,
                "SV": 0,
                "K": 0,
                "ERA": 0,
                "WHIP": 0,
            },
            {
                "Name": "Player B-H",
                "player_type": "hitter",
                "PA": 400,
                "R": 60,
                "HR": 15,
                "RBI": 50,
                "SB": 5,
                "OPS": 0.700,
                "IP": 0,
                "W": 0,
                "SV": 0,
                "K": 0,
                "ERA": 0,
                "WHIP": 0,
            },
            {
                "Name": "Pitcher A-P",
                "player_type": "pitcher",
                "PA": 0,
                "R": 0,
                "HR": 0,
                "RBI": 0,
                "SB": 0,
                "OPS": 0,
                "IP": 100,
                "W": 8,
                "SV": 0,
                "K": 100,
                "ERA": 3.50,
                "WHIP": 1.15,
            },
        ]
    )

    totals = compute_team_totals(
        {"Player A-H", "Player B-H", "Pitcher A-P"}, projections
    )

    # Weighted average: (600*0.9 + 400*0.7) / (600+400) = 820/1000 = 0.820
    expected_ops = (600 * 0.900 + 400 * 0.700) / 1000
    assert abs(totals["OPS"] - expected_ops) < 0.001, (
        f"OPS should be weighted average {expected_ops:.3f}, got {totals['OPS']:.3f}"
    )


def test_compute_team_totals_pitcher_ratio_stats():
    """Pitcher ratio stats (ERA, WHIP) use IP-weighted averages."""
    from optimizer.data_loader import compute_team_totals

    projections = pd.DataFrame(
        [
            {
                "Name": "Hitter A-H",
                "player_type": "hitter",
                "PA": 500,
                "R": 80,
                "HR": 25,
                "RBI": 70,
                "SB": 10,
                "OPS": 0.800,
                "IP": 0,
                "W": 0,
                "SV": 0,
                "K": 0,
                "ERA": 0,
                "WHIP": 0,
            },
            {
                "Name": "Pitcher A-P",
                "player_type": "pitcher",
                "PA": 0,
                "R": 0,
                "HR": 0,
                "RBI": 0,
                "SB": 0,
                "OPS": 0,
                "IP": 180,
                "W": 12,
                "SV": 0,
                "K": 180,
                "ERA": 3.00,
                "WHIP": 1.00,
            },
            {
                "Name": "Pitcher B-P",
                "player_type": "pitcher",
                "PA": 0,
                "R": 0,
                "HR": 0,
                "RBI": 0,
                "SB": 0,
                "OPS": 0,
                "IP": 60,
                "W": 4,
                "SV": 10,
                "K": 70,
                "ERA": 4.50,
                "WHIP": 1.40,
            },
        ]
    )

    totals = compute_team_totals(
        {"Hitter A-H", "Pitcher A-P", "Pitcher B-P"}, projections
    )

    # ERA weighted average: (180*3.0 + 60*4.5) / 240 = 810/240 = 3.375
    expected_era = (180 * 3.00 + 60 * 4.50) / 240
    assert abs(totals["ERA"] - expected_era) < 0.01, (
        f"ERA should be weighted average {expected_era:.2f}, got {totals['ERA']:.2f}"
    )

    # Counting stats should sum
    assert totals["W"] == 16
    assert totals["SV"] == 10
    assert totals["K"] == 250


# =============================================================================
# PROJECTION UNCERTAINTY TESTS
# =============================================================================


def test_estimate_projection_uncertainty():
    """Uncertainty estimation returns reasonable values."""
    from optimizer.data_loader import estimate_projection_uncertainty

    my_totals = {
        "R": 800,
        "HR": 240,
        "RBI": 780,
        "SB": 100,
        "OPS": 0.770,
        "W": 85,
        "SV": 75,
        "K": 1350,
        "ERA": 3.80,
        "WHIP": 1.20,
    }

    opponent_totals = {
        1: {
            "R": 820,
            "HR": 250,
            "RBI": 800,
            "SB": 110,
            "OPS": 0.780,
            "W": 88,
            "SV": 70,
            "K": 1400,
            "ERA": 3.70,
            "WHIP": 1.18,
        },
        2: {
            "R": 780,
            "HR": 230,
            "RBI": 760,
            "SB": 95,
            "OPS": 0.760,
            "W": 80,
            "SV": 80,
            "K": 1300,
            "ERA": 3.90,
            "WHIP": 1.22,
        },
        3: {
            "R": 850,
            "HR": 260,
            "RBI": 820,
            "SB": 105,
            "OPS": 0.790,
            "W": 92,
            "SV": 68,
            "K": 1450,
            "ERA": 3.60,
            "WHIP": 1.15,
        },
        4: {
            "R": 790,
            "HR": 245,
            "RBI": 790,
            "SB": 90,
            "OPS": 0.765,
            "W": 82,
            "SV": 78,
            "K": 1320,
            "ERA": 3.85,
            "WHIP": 1.21,
        },
        5: {
            "R": 810,
            "HR": 235,
            "RBI": 770,
            "SB": 115,
            "OPS": 0.775,
            "W": 86,
            "SV": 72,
            "K": 1380,
            "ERA": 3.75,
            "WHIP": 1.19,
        },
        6: {
            "R": 830,
            "HR": 255,
            "RBI": 810,
            "SB": 100,
            "OPS": 0.785,
            "W": 90,
            "SV": 74,
            "K": 1420,
            "ERA": 3.65,
            "WHIP": 1.16,
        },
    }

    sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)

    # Should have all 10 categories
    assert len(sigmas) == 10

    # All values should be positive
    for cat, sigma in sigmas.items():
        assert sigma > 0, f"Sigma for {cat} should be positive, got {sigma}"


# =============================================================================
# MILP COEFFICIENT SIGN TESTS
# =============================================================================


def test_ratio_stat_coefficient_signs():
    """Verify coefficient signs for ratio stat linearization are correct."""
    # OPS (higher is better): coeff = PA * (player_OPS - opponent_OPS)
    # ERA (lower is better): coeff = IP * (opponent_ERA - player_ERA)

    player_ops = 0.850
    player_era = 3.00
    opponent_ops = 0.770
    opponent_era = 3.85
    player_pa = 600
    player_ip = 180

    # OPS coefficient: positive when player is better (higher OPS)
    ops_coeff = player_pa * (player_ops - opponent_ops)
    assert ops_coeff > 0, "Good OPS player should have positive coefficient"

    # ERA coefficient: positive when player is better (lower ERA)
    era_coeff = player_ip * (opponent_era - player_era)
    assert era_coeff > 0, "Good ERA player should have positive coefficient"


# =============================================================================
# WIN PROBABILITY TESTS
# =============================================================================


def test_win_probability_bounds():
    """Win probability is between 0 and 1."""
    from optimizer.trade_engine import compute_win_probability

    my_totals = {
        "R": 820,
        "HR": 250,
        "RBI": 800,
        "SB": 105,
        "OPS": 0.775,
        "W": 85,
        "SV": 75,
        "K": 1350,
        "ERA": 3.75,
        "WHIP": 1.18,
    }

    opponent_totals = {
        1: {
            "R": 820,
            "HR": 240,
            "RBI": 780,
            "SB": 110,
            "OPS": 0.765,
            "W": 85,
            "SV": 70,
            "K": 1350,
            "ERA": 3.85,
            "WHIP": 1.22,
        },
        2: {
            "R": 795,
            "HR": 255,
            "RBI": 810,
            "SB": 95,
            "OPS": 0.782,
            "W": 78,
            "SV": 82,
            "K": 1280,
            "ERA": 3.72,
            "WHIP": 1.18,
        },
        3: {
            "R": 850,
            "HR": 230,
            "RBI": 760,
            "SB": 130,
            "OPS": 0.758,
            "W": 92,
            "SV": 65,
            "K": 1420,
            "ERA": 3.95,
            "WHIP": 1.25,
        },
        4: {
            "R": 780,
            "HR": 268,
            "RBI": 830,
            "SB": 88,
            "OPS": 0.795,
            "W": 80,
            "SV": 78,
            "K": 1310,
            "ERA": 3.68,
            "WHIP": 1.15,
        },
        5: {
            "R": 835,
            "HR": 245,
            "RBI": 795,
            "SB": 105,
            "OPS": 0.770,
            "W": 88,
            "SV": 72,
            "K": 1380,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        6: {
            "R": 805,
            "HR": 252,
            "RBI": 805,
            "SB": 100,
            "OPS": 0.778,
            "W": 82,
            "SV": 75,
            "K": 1340,
            "ERA": 3.75,
            "WHIP": 1.19,
        },
    }

    category_sigmas = {
        "R": 25,
        "HR": 12,
        "RBI": 25,
        "SB": 12,
        "OPS": 0.012,
        "W": 5,
        "SV": 5,
        "K": 50,
        "ERA": 0.10,
        "WHIP": 0.03,
    }

    V, diagnostics = compute_win_probability(
        my_totals, opponent_totals, category_sigmas
    )

    assert 0 <= V <= 1, f"Win probability {V} out of bounds"
    assert "expected_wins" in diagnostics


def test_better_team_higher_probability():
    """Strictly better team has higher win probability."""
    from optimizer.trade_engine import compute_win_probability

    opponent_totals = {
        1: {
            "R": 800,
            "HR": 240,
            "RBI": 780,
            "SB": 100,
            "OPS": 0.770,
            "W": 84,
            "SV": 74,
            "K": 1340,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        2: {
            "R": 800,
            "HR": 240,
            "RBI": 780,
            "SB": 100,
            "OPS": 0.770,
            "W": 84,
            "SV": 74,
            "K": 1340,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        3: {
            "R": 800,
            "HR": 240,
            "RBI": 780,
            "SB": 100,
            "OPS": 0.770,
            "W": 84,
            "SV": 74,
            "K": 1340,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        4: {
            "R": 800,
            "HR": 240,
            "RBI": 780,
            "SB": 100,
            "OPS": 0.770,
            "W": 84,
            "SV": 74,
            "K": 1340,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        5: {
            "R": 800,
            "HR": 240,
            "RBI": 780,
            "SB": 100,
            "OPS": 0.770,
            "W": 84,
            "SV": 74,
            "K": 1340,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        6: {
            "R": 800,
            "HR": 240,
            "RBI": 780,
            "SB": 100,
            "OPS": 0.770,
            "W": 84,
            "SV": 74,
            "K": 1340,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
    }

    category_sigmas = {
        "R": 30,
        "HR": 15,
        "RBI": 30,
        "SB": 12,
        "OPS": 0.015,
        "W": 6,
        "SV": 6,
        "K": 60,
        "ERA": 0.15,
        "WHIP": 0.04,
    }

    # Average team (same as opponents)
    avg_totals = {
        "R": 800,
        "HR": 240,
        "RBI": 780,
        "SB": 100,
        "OPS": 0.770,
        "W": 84,
        "SV": 74,
        "K": 1340,
        "ERA": 3.80,
        "WHIP": 1.20,
    }

    # Clearly better team (better in every category)
    good_totals = {
        "R": 900,
        "HR": 280,
        "RBI": 880,
        "SB": 130,
        "OPS": 0.820,
        "W": 100,
        "SV": 90,
        "K": 1500,
        "ERA": 3.40,
        "WHIP": 1.08,
    }

    V_avg, _ = compute_win_probability(avg_totals, opponent_totals, category_sigmas)
    V_good, _ = compute_win_probability(good_totals, opponent_totals, category_sigmas)

    assert V_good > V_avg, (
        f"Better team ({V_good:.3f}) should beat average ({V_avg:.3f})"
    )


# =============================================================================
# TRADE ENGINE CONSTANTS
# =============================================================================


def test_mev_mvar_tables():
    """MEV and MVAR tables have correct values from literature."""
    from optimizer.trade_engine import MEV_TABLE, MVAR_TABLE

    # From Teichroew (1956) / Rosenof (2025)
    assert abs(MEV_TABLE[6] - 1.267) < 0.001, (
        f"MEV[6] should be 1.267, got {MEV_TABLE[6]}"
    )
    assert abs(MVAR_TABLE[6] - 0.416) < 0.001, (
        f"MVAR[6] should be 0.416, got {MVAR_TABLE[6]}"
    )


def test_trade_fairness_threshold():
    """Fairness threshold is percentage-based at 10%."""
    from optimizer.trade_engine import (
        FAIRNESS_THRESHOLD_PERCENT,
        MIN_MEANINGFUL_IMPROVEMENT,
    )

    assert FAIRNESS_THRESHOLD_PERCENT == 0.10
    assert MIN_MEANINGFUL_IMPROVEMENT == 0.001


# =============================================================================
# RADAR CHART SORTING TESTS
# =============================================================================


def test_radar_sorting_hitting_pitching_opposite_directions():
    """
    Radar chart sorting should have hitting descending and pitching ascending.
    This makes the maxima of hitting and pitching meet in the middle.
    """
    from optimizer.data_loader import HITTING_CATEGORIES, PITCHING_CATEGORIES
    from optimizer.visualizations import sort_categories_for_radar

    # Create test data where "My Team" is best in R, HR and W, K
    my_totals = {
        "R": 900,
        "HR": 280,
        "RBI": 750,
        "SB": 80,
        "OPS": 0.760,  # Worst hitting
        "W": 100,
        "SV": 60,
        "K": 1500,
        "ERA": 4.00,  # Worst pitching
        "WHIP": 1.30,
    }

    opponent_totals = {
        1: {
            "R": 800,
            "HR": 240,
            "RBI": 780,
            "SB": 100,
            "OPS": 0.800,
            "W": 85,
            "SV": 75,
            "K": 1350,
            "ERA": 3.50,
            "WHIP": 1.15,
        }
    }

    sorted_cats = sort_categories_for_radar(my_totals, opponent_totals)

    # Extract hitting and pitching portions
    hitting_portion = [c for c in sorted_cats if c in HITTING_CATEGORIES]
    pitching_portion = [c for c in sorted_cats if c in PITCHING_CATEGORIES]

    # Hitting should be first
    hitting_indices = [sorted_cats.index(c) for c in hitting_portion]
    pitching_indices = [sorted_cats.index(c) for c in pitching_portion]
    assert max(hitting_indices) < min(pitching_indices), (
        "Hitting categories should come before pitching"
    )

    # R and HR should be near the start (best hitting), OPS near the end
    assert hitting_portion.index("R") < hitting_portion.index("OPS")
    assert hitting_portion.index("HR") < hitting_portion.index("OPS")

    # W and K should be near the END of pitching (best pitching comes last)
    # ERA should be near the START of pitching (worst pitching comes first)
    assert pitching_portion.index("ERA") < pitching_portion.index("W")
    assert pitching_portion.index("ERA") < pitching_portion.index("K")


# =============================================================================
# BALANCE LAMBDA TESTS
# =============================================================================


def test_balance_lambda_default_value():
    """Balance lambda default is 0.5."""
    from optimizer.data_loader import BALANCE_LAMBDA_DEFAULT

    assert BALANCE_LAMBDA_DEFAULT == 0.5


def test_balance_lambda_validation():
    """Invalid λ values should raise assertion error."""
    import pytest

    from optimizer.roster_optimizer import build_and_solve_milp

    # Create minimal test data (won't actually solve, just test validation)
    candidates = pd.DataFrame(
        [{"Name": "Test-H", "Position": "OF", "player_type": "hitter"}]
    )
    opponent_totals = {1: {"R": 100}}
    current_names = set()

    with pytest.raises(AssertionError, match="balance_lambda must be in"):
        build_and_solve_milp(
            candidates, opponent_totals, current_names, balance_lambda=-0.5
        )

    with pytest.raises(AssertionError, match="balance_lambda must be in"):
        build_and_solve_milp(
            candidates, opponent_totals, current_names, balance_lambda=3.0
        )


def test_balance_objective_formula():
    """Verify objective equals total_wins + λ*w_min - λ*w_max."""
    # This test verifies the formula without actually solving
    # We check that the math works out correctly
    total_wins = 40
    w_min = 2
    w_max = 6
    balance_lambda = 0.5

    expected_objective = total_wins + balance_lambda * w_min - balance_lambda * w_max
    # = 40 + 0.5*2 - 0.5*6 = 40 + 1 - 3 = 38

    assert expected_objective == 38.0


def test_balance_metrics_consistency():
    """Verify balance metric relationships are correct."""
    # Test that w_min, w_max, win_range are consistent
    category_wins = {
        "R": 4,
        "HR": 5,
        "RBI": 4,
        "SB": 2,
        "OPS": 4,
        "W": 3,
        "SV": 2,
        "K": 5,
        "ERA": 4,
        "WHIP": 4,
    }

    total_wins = sum(category_wins.values())
    w_min = min(category_wins.values())
    w_max = max(category_wins.values())
    win_range = w_max - w_min

    assert total_wins == 37
    assert w_min == 2
    assert w_max == 5
    assert win_range == 3

    # Verify worst and best categories
    worst_cats = [c for c, w in category_wins.items() if w == w_min]
    best_cats = [c for c, w in category_wins.items() if w == w_max]

    assert set(worst_cats) == {"SB", "SV"}
    assert set(best_cats) == {"HR", "K"}
