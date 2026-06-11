"""
Minimal test suite per AGENTS.md: no classes, no fixtures, no mocking.
Each test is self-contained with inline test data.
"""

import numpy as np
import pandas as pd

from optimizer.lineup_solver import compute_totals_for_starters
from optimizer.player_scoring import add_fantasy_value, add_mew
from optimizer.players import get_eligible_slots
from optimizer.swap_evaluator import add_bench_value, compute_exact_msv
from optimizer.win_model import (
    compute_ew_gradient,
    compute_win_probability,
    estimate_projection_uncertainty,
)


def _make_hitter(
    name: str,
    pa: float,
    r: float,
    hr: float,
    rbi: float,
    sb: float,
    ops: float,
    war: float = 2.0,
    owner: str | None = None,
    position: str = "OF",
    roster_status: str | None = None,
) -> dict:
    """Helper to build a hitter row."""
    return {
        "Name": name,
        "Team": "NYY",
        "Position": position,
        "player_type": "hitter",
        "PA": pa,
        "IP": 0.0,
        "R": r,
        "HR": hr,
        "RBI": rbi,
        "SB": sb,
        "OPS": ops,
        "W": 0.0,
        "SV": 0.0,
        "K": 0.0,
        "ERA": 0.0,
        "WHIP": 0.0,
        "WAR": war,
        "owner": owner,
        "roster_status": roster_status,
    }


def _make_pitcher(
    name: str,
    ip: float,
    w: float,
    sv: float,
    k: float,
    era: float,
    whip: float,
    war: float = 2.0,
    owner: str | None = None,
    position: str = "SP",
    roster_status: str | None = None,
) -> dict:
    """Helper to build a pitcher row."""
    return {
        "Name": name,
        "Team": "LAD",
        "Position": position,
        "player_type": "pitcher",
        "PA": 0.0,
        "IP": ip,
        "R": 0.0,
        "HR": 0.0,
        "RBI": 0.0,
        "SB": 0.0,
        "OPS": 0.0,
        "W": w,
        "SV": sv,
        "K": k,
        "ERA": era,
        "WHIP": whip,
        "WAR": war,
        "owner": owner,
        "roster_status": roster_status,
    }


def _synthetic_totals():
    """Create synthetic my_totals and opponent_totals for gradient tests."""
    my_totals = {
        "R": 800.0,
        "HR": 250.0,
        "RBI": 780.0,
        "SB": 100.0,
        "OPS": 0.770,
        "W": 80.0,
        "SV": 45.0,
        "K": 1200.0,
        "ERA": 3.80,
        "WHIP": 1.20,
        "PA": 5500.0,
        "IP": 1200.0,
    }
    opponent_totals = {
        1: {
            "R": 790,
            "HR": 240,
            "RBI": 770,
            "SB": 110,
            "OPS": 0.760,
            "W": 75,
            "SV": 50,
            "K": 1180,
            "ERA": 3.90,
            "WHIP": 1.22,
        },
        2: {
            "R": 810,
            "HR": 260,
            "RBI": 800,
            "SB": 90,
            "OPS": 0.780,
            "W": 85,
            "SV": 40,
            "K": 1220,
            "ERA": 3.70,
            "WHIP": 1.18,
        },
    }
    return my_totals, opponent_totals


def test_ew_gradient_sign_convention():
    """g_c > 0 for C⁺, g_c < 0 for C⁻."""
    my_totals, opponent_totals = _synthetic_totals()
    sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)

    positive_cats = {"R", "HR", "RBI", "SB", "OPS", "W", "SV", "K"}
    negative_cats = {"ERA", "WHIP"}

    for cat in positive_cats:
        assert gradient[cat] > 0, (
            f"Gradient for {cat} should be positive (C⁺), got {gradient[cat]:.6f}"
        )
    for cat in negative_cats:
        assert gradient[cat] < 0, (
            f"Gradient for {cat} should be negative (C⁻), got {gradient[cat]:.6f}"
        )


def test_mew_era_sign_check():
    """Low-ERA pitcher must have higher MEW than high-ERA pitcher."""
    my_totals, opponent_totals = _synthetic_totals()
    sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)

    rows = [
        _make_pitcher("LowERA-P", ip=180, w=12, sv=0, k=180, era=2.50, whip=1.00),
        _make_pitcher("HighERA-P", ip=180, w=12, sv=0, k=180, era=4.50, whip=1.40),
        _make_hitter("Filler-H", pa=500, r=80, hr=25, rbi=80, sb=10, ops=0.800),
    ]
    players = pd.DataFrame(rows)
    players = add_mew(players, my_totals, gradient)

    low_era_mew = players.loc[players["Name"] == "LowERA-P", "MEW"].iloc[0]
    high_era_mew = players.loc[players["Name"] == "HighERA-P", "MEW"].iloc[0]

    assert low_era_mew > high_era_mew, (
        f"Low-ERA pitcher MEW ({low_era_mew:.4f}) should be > "
        f"high-ERA pitcher MEW ({high_era_mew:.4f}). Sign error in MEW formula."
    )


def test_mew_unified_formula():
    """MEW formula produces correct results without hitter/pitcher branching."""
    my_totals, opponent_totals = _synthetic_totals()
    sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)

    rows = [
        _make_hitter("Hitter-H", pa=600, r=90, hr=30, rbi=90, sb=15, ops=0.820),
        _make_pitcher("Pitcher-P", ip=200, w=15, sv=0, k=200, era=3.00, whip=1.05),
    ]
    players = pd.DataFrame(rows)
    players = add_mew(players, my_totals, gradient)

    hitter_mew = players.loc[players["Name"] == "Hitter-H", "MEW"].iloc[0]
    pitcher_mew = players.loc[players["Name"] == "Pitcher-P", "MEW"].iloc[0]

    # Hitter: IP=0 so pitching terms vanish, only hitting terms contribute
    h = players.loc[players["Name"] == "Hitter-H"].iloc[0]
    expected_h = (
        gradient["R"] * h["R"]
        + gradient["HR"] * h["HR"]
        + gradient["RBI"] * h["RBI"]
        + gradient["SB"] * h["SB"]
        + gradient["W"] * h["W"]
        + gradient["SV"] * h["SV"]
        + gradient["K"] * h["K"]
        + gradient["OPS"] * h["PA"] * (h["OPS"] - my_totals["OPS"]) / my_totals["PA"]
        + gradient["ERA"] * h["IP"] * (h["ERA"] - my_totals["ERA"]) / my_totals["IP"]
        + gradient["WHIP"] * h["IP"] * (h["WHIP"] - my_totals["WHIP"]) / my_totals["IP"]
    )
    assert abs(hitter_mew - expected_h) < 1e-10, (
        f"Hitter MEW {hitter_mew:.6f} != expected {expected_h:.6f}"
    )

    # Pitcher: PA=0 so hitting terms vanish, only pitching terms contribute
    p = players.loc[players["Name"] == "Pitcher-P"].iloc[0]
    expected_p = (
        gradient["R"] * p["R"]
        + gradient["HR"] * p["HR"]
        + gradient["RBI"] * p["RBI"]
        + gradient["SB"] * p["SB"]
        + gradient["W"] * p["W"]
        + gradient["SV"] * p["SV"]
        + gradient["K"] * p["K"]
        + gradient["OPS"] * p["PA"] * (p["OPS"] - my_totals["OPS"]) / my_totals["PA"]
        + gradient["ERA"] * p["IP"] * (p["ERA"] - my_totals["ERA"]) / my_totals["IP"]
        + gradient["WHIP"] * p["IP"] * (p["WHIP"] - my_totals["WHIP"]) / my_totals["IP"]
    )
    assert abs(pitcher_mew - expected_p) < 1e-10, (
        f"Pitcher MEW {pitcher_mew:.6f} != expected {expected_p:.6f}"
    )


def test_ratio_stat_delta_trap():
    """Replacing below-average-ERA pitcher with fewer IP can worsen ERA."""
    # Team: 1000 IP, 3.00 ERA
    # Remove: 200 IP, 2.80 ERA. Add: 50 IP, 2.50 ERA.
    # ΔERA ≈ [50×(2.50−3.00) − 200×(2.80−3.00)] / 1000 = +0.015 (worsens)
    delta_era = (50 * (2.50 - 3.00) - 200 * (2.80 - 3.00)) / 1000
    assert delta_era > 0, (
        f"Expected positive ΔERA (worsening), got {delta_era:.6f}. "
        f"The volume loss dominates the rate improvement."
    )
    assert abs(delta_era - 0.015) < 0.001, f"Expected ΔERA ≈ 0.015, got {delta_era:.6f}"


def test_team_totals_weighted_average():
    """ERA/OPS must be IP/PA-weighted averages, not sums."""
    rows = [
        _make_pitcher("P1-P", ip=100, w=8, sv=0, k=100, era=3.00, whip=1.10),
        _make_pitcher("P2-P", ip=50, w=4, sv=0, k=50, era=4.00, whip=1.30),
        _make_hitter("H1-H", pa=500, r=70, hr=20, rbi=70, sb=10, ops=0.800),
        _make_hitter("H2-H", pa=300, r=40, hr=10, rbi=40, sb=5, ops=0.700),
    ]
    players = pd.DataFrame(rows)
    totals = compute_totals_for_starters({"P1-P", "P2-P", "H1-H", "H2-H"}, players)

    expected_era = (100 * 3.00 + 50 * 4.00) / 150
    assert abs(totals["ERA"] - expected_era) < 1e-10, (
        f"ERA should be {expected_era:.4f} (weighted avg), got {totals['ERA']:.4f}. "
        f"ERA must NOT be summed (7.0 would indicate summation)."
    )

    expected_ops = (500 * 0.800 + 300 * 0.700) / 800
    assert abs(totals["OPS"] - expected_ops) < 1e-10, (
        f"OPS should be {expected_ops:.4f} (weighted avg), got {totals['OPS']:.4f}"
    )

    assert totals["PA"] == 800.0, f"PA should be 800, got {totals['PA']}"
    assert totals["IP"] == 150.0, f"IP should be 150, got {totals['IP']}"


def test_msv_identity_swap():
    """Swapping a player with themselves → MSV = 0."""
    rows = [
        _make_hitter(
            "H1-H",
            pa=500,
            r=70,
            hr=20,
            rbi=70,
            sb=10,
            ops=0.800,
            owner="The Big Dumpers",
            position="1B",
            roster_status="active",
        ),
        _make_hitter(
            "H2-H",
            pa=400,
            r=60,
            hr=15,
            rbi=60,
            sb=8,
            ops=0.750,
            owner="The Big Dumpers",
            position="2B",
            roster_status="active",
        ),
        _make_hitter(
            "H3-H",
            pa=450,
            r=65,
            hr=18,
            rbi=65,
            sb=12,
            ops=0.770,
            owner="The Big Dumpers",
            position="SS",
            roster_status="active",
        ),
        _make_hitter(
            "H4-H",
            pa=550,
            r=80,
            hr=25,
            rbi=80,
            sb=6,
            ops=0.810,
            owner="The Big Dumpers",
            position="3B",
            roster_status="active",
        ),
        _make_hitter(
            "H5-H",
            pa=480,
            r=75,
            hr=22,
            rbi=75,
            sb=20,
            ops=0.790,
            owner="The Big Dumpers",
            position="OF",
            roster_status="active",
        ),
        _make_hitter(
            "H6-H",
            pa=480,
            r=70,
            hr=20,
            rbi=70,
            sb=15,
            ops=0.780,
            owner="The Big Dumpers",
            position="OF",
            roster_status="active",
        ),
        _make_hitter(
            "H7-H",
            pa=480,
            r=70,
            hr=20,
            rbi=70,
            sb=15,
            ops=0.780,
            owner="The Big Dumpers",
            position="OF",
            roster_status="active",
        ),
        _make_hitter(
            "H8-H",
            pa=480,
            r=70,
            hr=20,
            rbi=70,
            sb=15,
            ops=0.780,
            owner="The Big Dumpers",
            position="OF",
            roster_status="active",
        ),
        _make_hitter(
            "H9-H",
            pa=480,
            r=70,
            hr=20,
            rbi=70,
            sb=15,
            ops=0.780,
            owner="The Big Dumpers",
            position="OF",
            roster_status="active",
        ),
        _make_hitter(
            "H10-H",
            pa=400,
            r=55,
            hr=12,
            rbi=55,
            sb=5,
            ops=0.730,
            owner="The Big Dumpers",
            position="C",
            roster_status="active",
        ),
        _make_hitter(
            "H11-H",
            pa=350,
            r=45,
            hr=10,
            rbi=45,
            sb=3,
            ops=0.710,
            owner="The Big Dumpers",
            position="DH",
            roster_status="active",
        ),
        _make_pitcher(
            "P1-P",
            ip=180,
            w=12,
            sv=0,
            k=180,
            era=3.20,
            whip=1.10,
            owner="The Big Dumpers",
            position="SP",
            roster_status="active",
        ),
        _make_pitcher(
            "P2-P",
            ip=170,
            w=11,
            sv=0,
            k=170,
            era=3.40,
            whip=1.15,
            owner="The Big Dumpers",
            position="SP",
            roster_status="active",
        ),
        _make_pitcher(
            "P3-P",
            ip=160,
            w=10,
            sv=0,
            k=160,
            era=3.60,
            whip=1.18,
            owner="The Big Dumpers",
            position="SP",
            roster_status="active",
        ),
        _make_pitcher(
            "P4-P",
            ip=150,
            w=9,
            sv=0,
            k=150,
            era=3.80,
            whip=1.20,
            owner="The Big Dumpers",
            position="SP",
            roster_status="active",
        ),
        _make_pitcher(
            "P5-P",
            ip=140,
            w=8,
            sv=0,
            k=140,
            era=4.00,
            whip=1.25,
            owner="The Big Dumpers",
            position="SP",
            roster_status="active",
        ),
        _make_pitcher(
            "P6-P",
            ip=70,
            w=3,
            sv=25,
            k=70,
            era=3.50,
            whip=1.15,
            owner="The Big Dumpers",
            position="RP",
            roster_status="active",
        ),
        _make_pitcher(
            "P7-P",
            ip=65,
            w=2,
            sv=20,
            k=65,
            era=3.70,
            whip=1.20,
            owner="The Big Dumpers",
            position="RP",
            roster_status="active",
        ),
        # Bench
        _make_hitter(
            "BH1-H",
            pa=300,
            r=40,
            hr=10,
            rbi=40,
            sb=5,
            ops=0.720,
            owner="The Big Dumpers",
            position="OF",
            roster_status="active",
        ),
        _make_hitter(
            "BH2-H",
            pa=250,
            r=35,
            hr=8,
            rbi=35,
            sb=3,
            ops=0.700,
            owner="The Big Dumpers",
            position="1B",
            roster_status="active",
        ),
        _make_pitcher(
            "BP1-P",
            ip=100,
            w=5,
            sv=0,
            k=90,
            era=4.20,
            whip=1.30,
            owner="The Big Dumpers",
            position="SP",
            roster_status="active",
        ),
    ]

    # Need 28 total — pad with more bench
    for i in range(7):
        rows.append(
            _make_hitter(
                f"Bench{i}-H",
                pa=200,
                r=25,
                hr=5,
                rbi=25,
                sb=2,
                ops=0.680,
                owner="The Big Dumpers",
                position="OF",
                roster_status="active",
            )
        )

    # Add opponents and FAs for completeness
    for opp_idx in range(6):
        opp_name = f"Opp{opp_idx + 1}"
        for j in range(28):
            if j < 11:
                rows.append(
                    _make_hitter(
                        f"{opp_name}_H{j}-H",
                        pa=450,
                        r=60,
                        hr=18,
                        rbi=60,
                        sb=8,
                        ops=0.760,
                        owner=opp_name,
                        position=[
                            "C",
                            "1B",
                            "2B",
                            "SS",
                            "3B",
                            "OF",
                            "OF",
                            "OF",
                            "OF",
                            "OF",
                            "DH",
                        ][j],
                        roster_status="active",
                    )
                )
            else:
                rows.append(
                    _make_pitcher(
                        f"{opp_name}_P{j}-P",
                        ip=120,
                        w=7,
                        sv=3,
                        k=110,
                        era=3.90,
                        whip=1.22,
                        owner=opp_name,
                        position="SP" if j < 23 else "RP",
                        roster_status="active",
                    )
                )

    players = pd.DataFrame(rows)
    players = add_fantasy_value(players)

    my_roster = set(players[players["owner"] == "The Big Dumpers"]["Name"])
    assert len(my_roster) == 28, f"Expected 28 roster players, got {len(my_roster)}"

    # Use actual opponent totals from the test data for consistent EW
    from optimizer.lineup_solver import solve_lineup

    opp_totals: dict[int, dict[str, float]] = {}
    opp_teams = sorted(
        t
        for t in players[players["owner"].notna()]["owner"].unique()
        if t != "The Big Dumpers"
    )
    for i, team in enumerate(opp_teams):
        opp_roster = set(players[players["owner"] == team]["Name"])
        opp_lineup = solve_lineup(opp_roster, players, "FV")
        opp_totals[i + 1] = compute_totals_for_starters(set(opp_lineup.keys()), players)

    # Initial FV lineup to bootstrap gradient/MEW
    fv_lineup = solve_lineup(my_roster, players, "FV")
    my_totals = compute_totals_for_starters(set(fv_lineup.keys()), players)
    sigmas = estimate_projection_uncertainty(my_totals, opp_totals)
    gradient = compute_ew_gradient(my_totals, opp_totals, sigmas)
    players = add_mew(players, my_totals, gradient)

    # Re-solve with MEW (matches what compute_exact_msv does internally)
    mew_lineup = solve_lineup(my_roster, players, "MEW")
    mew_totals = compute_totals_for_starters(set(mew_lineup.keys()), players)
    actual_ew, _ = compute_win_probability(mew_totals, opp_totals, sigmas)

    # "Swap" H1-H with H1-H (identity)
    result = compute_exact_msv(
        {"H1-H"},
        {"H1-H"},
        my_roster,
        players,
        opp_totals,
        sigmas,
        current_ew=actual_ew,
    )
    assert abs(result["msv"]) < 1e-10, (
        f"Identity swap should have MSV=0, got {result['msv']:.6f}"
    )


def test_pv_constraint_filters_correctly():
    """Trades where PV(send) − PV(receive) < −ε must be excluded."""
    from optimizer.trade_finder import evaluate_trade

    rows = [
        {
            **_make_hitter(
                "MyGuy-H",
                pa=500,
                r=70,
                hr=20,
                rbi=70,
                sb=10,
                ops=0.800,
                owner="The Big Dumpers",
                roster_status="active",
                position="1B",
            ),
            "PV": 1.0,
            "FV": 2.0,
            "MEW": 1.5,
        },
        {
            **_make_hitter(
                "TheirStar-H",
                pa=600,
                r=90,
                hr=30,
                rbi=90,
                sb=15,
                ops=0.850,
                owner="OppTeam",
                roster_status="active",
                position="1B",
            ),
            "PV": 5.0,
            "FV": 4.0,
            "MEW": 3.0,
        },
    ]
    players = pd.DataFrame(rows)

    my_roster = {"MyGuy-H"}
    opp_roster = {"TheirStar-H"}

    # PV(send=MyGuy, 1.0) - PV(recv=TheirStar, 5.0) = -4.0 < -0.10
    result = evaluate_trade(
        send_names={"MyGuy-H"},
        receive_names={"TheirStar-H"},
        my_roster_names=my_roster,
        opponent_roster_names=opp_roster,
        trade_opponent_id=1,
        players=players,
        opponent_totals={
            1: {
                "R": 800,
                "HR": 250,
                "RBI": 780,
                "SB": 100,
                "OPS": 0.770,
                "W": 80,
                "SV": 45,
                "K": 1200,
                "ERA": 3.80,
                "WHIP": 1.20,
            }
        },
        category_sigmas={
            "R": 50,
            "HR": 22,
            "RBI": 50,
            "SB": 15,
            "OPS": 0.012,
            "W": 10,
            "SV": 9,
            "K": 72,
            "ERA": 0.3,
            "WHIP": 0.05,
        },
        current_ew=30.0,
        current_total_bv=0.0,
        pv_max_loss_frac=0.10,
    )

    assert not result["pv_feasible"], (
        f"Trade should be PV-infeasible (balance={result['pv_balance']:.2f}), "
        f"but pv_feasible={result['pv_feasible']}"
    )


def test_gradient_based_bv_position_aware():
    """Bench player eligible for high-absence slot should have higher BV."""
    my_totals, opponent_totals = _synthetic_totals()
    sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)

    rows = [
        _make_hitter(
            "Starter_C-H",
            pa=400,
            r=50,
            hr=15,
            rbi=50,
            sb=3,
            ops=0.730,
            owner="The Big Dumpers",
            position="C",
            roster_status="active",
        ),
        _make_hitter(
            "Starter_UTIL-H",
            pa=500,
            r=70,
            hr=20,
            rbi=70,
            sb=10,
            ops=0.780,
            owner="The Big Dumpers",
            position="DH",
            roster_status="active",
        ),
        # Bench players with identical MEW-relevant stats but different positions
        _make_hitter(
            "Bench_C-H",
            pa=300,
            r=35,
            hr=10,
            rbi=35,
            sb=2,
            ops=0.710,
            owner="The Big Dumpers",
            position="C",
            roster_status="active",
        ),
        _make_hitter(
            "Bench_UTIL-H",
            pa=300,
            r=35,
            hr=10,
            rbi=35,
            sb=2,
            ops=0.710,
            owner="The Big Dumpers",
            position="DH",
            roster_status="active",
        ),
        # FA
        _make_hitter("FA_low-H", pa=200, r=20, hr=5, rbi=20, sb=1, ops=0.650),
    ]

    players = pd.DataFrame(rows)
    players = add_mew(players, my_totals, gradient)

    # my_lineup: only 2 starters for this minimal test
    my_lineup = {"Starter_C-H": "C", "Starter_UTIL-H": "UTIL"}

    my_roster_names = {"Starter_C-H", "Starter_UTIL-H", "Bench_C-H", "Bench_UTIL-H"}
    players = add_bench_value(players, my_lineup, my_roster_names)

    bv_c = players.loc[players["Name"] == "Bench_C-H", "BV"].iloc[0]
    bv_util = players.loc[players["Name"] == "Bench_UTIL-H", "BV"].iloc[0]

    # C absence rate = 0.25, UTIL absence rate = 0.15
    assert bv_c > bv_util, (
        f"C-eligible bench player BV ({bv_c:.4f}) should be > "
        f"UTIL-eligible bench player BV ({bv_util:.4f}) "
        f"because C has higher absence rate (0.25 vs 0.15)."
    )


def test_mew_lineup_differs_from_fv_lineup():
    """My lineup using MEW should differ from FV when gradient is non-uniform."""
    from optimizer.lineup_solver import solve_lineup

    # Gradient heavily weights SB
    gradient = {
        "R": 0.01,
        "HR": 0.01,
        "RBI": 0.01,
        "SB": 5.0,
        "OPS": 0.01,
        "W": 0.01,
        "SV": 0.01,
        "K": 0.01,
        "ERA": -0.01,
        "WHIP": -0.01,
    }
    my_totals = {
        "R": 800,
        "HR": 250,
        "RBI": 780,
        "SB": 50,
        "OPS": 0.770,
        "W": 80,
        "SV": 45,
        "K": 1200,
        "ERA": 3.80,
        "WHIP": 1.20,
        "PA": 5500,
        "IP": 1200,
    }

    # Two UTIL-eligible hitters: one has high FV but low SB, other lower FV but high SB
    # Include filler pitchers so FV z-score computation doesn't get NaN std
    rows = [
        _make_hitter(
            "HighFV-H",
            pa=600,
            r=100,
            hr=40,
            rbi=100,
            sb=5,
            ops=0.900,
            owner="The Big Dumpers",
            position="DH",
            roster_status="active",
        ),
        _make_hitter(
            "HighSB-H",
            pa=500,
            r=60,
            hr=10,
            rbi=50,
            sb=50,
            ops=0.700,
            owner="The Big Dumpers",
            position="DH",
            roster_status="active",
        ),
        _make_pitcher("FillerP1-P", ip=180, w=12, sv=0, k=180, era=3.20, whip=1.10),
        _make_pitcher("FillerP2-P", ip=100, w=6, sv=10, k=100, era=4.50, whip=1.35),
    ]
    players = pd.DataFrame(rows)
    players = add_fantasy_value(players)
    players = add_mew(players, my_totals, gradient)

    # FV lineup: HighFV wins (higher FV)
    fv_lineup = solve_lineup({"HighFV-H", "HighSB-H"}, players, "FV")
    # MEW lineup: HighSB wins (gradient heavily weights SB)
    mew_lineup = solve_lineup({"HighFV-H", "HighSB-H"}, players, "MEW")

    assert "HighFV-H" in fv_lineup, "FV should prefer HighFV player"
    assert "HighSB-H" in mew_lineup, "MEW should prefer HighSB player"

    # Both can start since there's UTIL slot, but if only 1 slot available,
    # MEW should pick the SB specialist
    assert (
        players.loc[players["Name"] == "HighFV-H", "FV"].iloc[0]
        > players.loc[players["Name"] == "HighSB-H", "FV"].iloc[0]
    ), "HighFV should have higher FV"
    assert (
        players.loc[players["Name"] == "HighSB-H", "MEW"].iloc[0]
        > players.loc[players["Name"] == "HighFV-H", "MEW"].iloc[0]
    ), "HighSB should have higher MEW when gradient heavily weights SB"
