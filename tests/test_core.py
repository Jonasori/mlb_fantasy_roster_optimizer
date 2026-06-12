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


# ============================================================================
# Banked YTD integration (full-season standings = banked + rest-of-season)
# ============================================================================


def test_blend_season_totals_counting_and_ratio():
    """Counting stats sum; ratio stats blend by PA/IP weight."""
    from optimizer.lineup_solver import blend_season_totals

    banked = {
        "R": 400.0,
        "HR": 100.0,
        "RBI": 380.0,
        "SB": 50.0,
        "W": 40.0,
        "SV": 20.0,
        "K": 600.0,
        "OPS": 0.720,
        "ERA": 3.50,
        "WHIP": 1.25,
        "PA": 3000.0,
        "IP": 600.0,
    }
    ros = {
        "R": 300.0,
        "HR": 80.0,
        "RBI": 300.0,
        "SB": 40.0,
        "W": 35.0,
        "SV": 18.0,
        "K": 500.0,
        "OPS": 0.780,
        "ERA": 3.20,
        "WHIP": 1.15,
        "PA": 2000.0,
        "IP": 500.0,
    }
    s = blend_season_totals(banked, ros)

    assert s["R"] == 700.0, f"counting R should sum to 700, got {s['R']}"
    assert s["HR"] == 180.0
    assert s["PA"] == 5000.0
    assert s["IP"] == 1100.0
    exp_ops = (3000.0 * 0.720 + 2000.0 * 0.780) / 5000.0
    assert abs(s["OPS"] - exp_ops) < 1e-9, f"OPS blend wrong: {s['OPS']} vs {exp_ops}"
    exp_era = (600.0 * 3.50 + 500.0 * 3.20) / 1100.0
    assert abs(s["ERA"] - exp_era) < 1e-9, f"ERA blend wrong: {s['ERA']} vs {exp_era}"


def test_blend_zero_banked_is_identity():
    """Zero banked totals (no games played) → season == rest-of-season."""
    from optimizer.lineup_solver import blend_season_totals

    zero = {
        "R": 0.0,
        "HR": 0.0,
        "RBI": 0.0,
        "SB": 0.0,
        "W": 0.0,
        "SV": 0.0,
        "K": 0.0,
        "OPS": 0.0,
        "ERA": 0.0,
        "WHIP": 0.0,
        "PA": 0.0,
        "IP": 0.0,
    }
    ros = {
        "R": 300.0,
        "HR": 80.0,
        "RBI": 300.0,
        "SB": 40.0,
        "W": 35.0,
        "SV": 18.0,
        "K": 500.0,
        "OPS": 0.780,
        "ERA": 3.20,
        "WHIP": 1.15,
        "PA": 2000.0,
        "IP": 500.0,
    }
    s = blend_season_totals(zero, ros)
    for cat in ("R", "HR", "OPS", "ERA", "WHIP", "PA", "IP"):
        assert abs(s[cat] - ros[cat]) < 1e-9, (
            f"zero-banked blend changed {cat}: {s[cat]} vs {ros[cat]}"
        )


def test_sigma_sqrt_f_scaling_and_f1_regression():
    """σ scales as √f: counting ÷√f, ratio ×√f; f=1 reproduces legacy."""
    my_totals, opponent_totals = _synthetic_totals()

    legacy = estimate_projection_uncertainty(my_totals, opponent_totals)
    f1 = estimate_projection_uncertainty(my_totals, opponent_totals, 1.0)
    quarter = estimate_projection_uncertainty(my_totals, opponent_totals, 0.25)

    for cat in legacy:
        assert abs(legacy[cat] - f1[cat]) < 1e-9, (
            f"f=1 must reproduce legacy σ for {cat}: {f1[cat]} vs {legacy[cat]}"
        )

    # Counting (HR): 1/√0.25 = 2.0 → doubled.
    assert abs(quarter["HR"] - legacy["HR"] * 2.0) < 1e-6, (
        f"counting σ should scale by 1/√f: {quarter['HR']} vs {legacy['HR'] * 2}"
    )
    # Ratio (OPS): √0.25 = 0.5 → halved.
    assert abs(quarter["OPS"] - legacy["OPS"] * 0.5) < 1e-9, (
        f"ratio σ should scale by √f: {quarter['OPS']} vs {legacy['OPS'] * 0.5}"
    )


def test_banked_deficit_lowers_beat_probability():
    """A banked category deficit must reduce the modeled beat probability."""
    from optimizer.lineup_solver import blend_season_totals

    sigmas = {
        "R": 50.0,
        "HR": 13.0,
        "RBI": 50.0,
        "SB": 15.0,
        "OPS": 0.012,
        "W": 10.0,
        "SV": 9.0,
        "K": 72.0,
        "ERA": 0.30,
        "WHIP": 0.05,
    }
    ros = {
        "R": 300.0,
        "HR": 140.0,
        "RBI": 300.0,
        "SB": 40.0,
        "W": 35.0,
        "SV": 18.0,
        "K": 500.0,
        "OPS": 0.770,
        "ERA": 3.20,
        "WHIP": 1.15,
        "PA": 2400.0,
        "IP": 500.0,
    }
    my_banked = {
        "R": 350.0,
        "HR": 100.0,
        "RBI": 350.0,
        "SB": 45.0,
        "W": 38.0,
        "SV": 20.0,
        "K": 560.0,
        "OPS": 0.760,
        "ERA": 3.30,
        "WHIP": 1.18,
        "PA": 2800.0,
        "IP": 560.0,
    }
    # Opponent identical except 20 more banked HR.
    opp_banked = dict(my_banked)
    opp_banked["HR"] = my_banked["HR"] + 20.0

    my_season = blend_season_totals(my_banked, ros)
    opp_season = blend_season_totals(opp_banked, ros)

    _, diag = compute_win_probability(my_season, {1: opp_season}, sigmas)
    hr_beat = diag["beat_probs"]["HR"][1]
    assert hr_beat < 0.5, (
        f"trailing by 20 banked HR (equal ros) must give beat prob < 0.5, "
        f"got {hr_beat:.3f}"
    )

    # Without banked (ros-only, equal): beat prob is exactly 0.5.
    _, diag0 = compute_win_probability(ros, {1: ros}, sigmas)
    assert abs(diag0["beat_probs"]["HR"][1] - 0.5) < 1e-9


def test_standings_to_banked_rejects_garbage():
    """Out-of-range rate values (mis-parse) → None (safe fallback)."""
    import pandas as pd

    from optimizer.banked import standings_to_banked_totals

    good = pd.DataFrame(
        [
            {
                "team_name": "A",
                "r": 400,
                "hr": 100,
                "rbi": 380,
                "sb": 50,
                "ops": 0.760,
                "w": 40,
                "sv": 20,
                "k": 600,
                "era": 3.5,
                "whip": 1.2,
            },
            {
                "team_name": "B",
                "r": 420,
                "hr": 110,
                "rbi": 400,
                "sb": 40,
                "ops": 0.740,
                "w": 42,
                "sv": 18,
                "k": 620,
                "era": 3.7,
                "whip": 1.25,
            },
        ]
    )
    out = standings_to_banked_totals(good)
    assert out is not None and "A" in out and out["A"]["HR"] == 100.0

    # OPS of 7.0 means the parser grabbed roto points, not OPS → reject.
    bad = good.copy()
    bad.loc[0, "ops"] = 7.0
    assert standings_to_banked_totals(bad) is None


# ============================================================================
# Injury-aware protection (the "only healthy catcher" bug)
# ============================================================================


def test_critical_slots_injury_aware():
    """Sole STARTABLE catcher is protected even if an IL catcher remains."""
    from optimizer.swap_evaluator import compute_critical_slots, find_protected_players

    rows = [
        {
            **_make_hitter(
                "HealthyC-H",
                pa=300,
                r=30,
                hr=8,
                rbi=30,
                sb=2,
                ops=0.700,
                owner="The Big Dumpers",
                position="C",
                roster_status="active",
            ),
            "injury_status": None,
        },
        {
            **_make_hitter(
                "InjuredC-H",
                pa=350,
                r=45,
                hr=15,
                rbi=45,
                sb=1,
                ops=0.800,
                owner="The Big Dumpers",
                position="C",
                roster_status="IR",
            ),
            "injury_status": "IL",
        },
        {
            **_make_hitter(
                "OF1-H",
                pa=500,
                r=70,
                hr=20,
                rbi=70,
                sb=10,
                ops=0.780,
                owner="The Big Dumpers",
                position="OF",
                roster_status="active",
            ),
            "injury_status": None,
        },
        {
            **_make_hitter(
                "OF2-H",
                pa=500,
                r=70,
                hr=20,
                rbi=70,
                sb=10,
                ops=0.780,
                owner="The Big Dumpers",
                position="OF",
                roster_status="active",
            ),
            "injury_status": None,
        },
    ]
    players = pd.DataFrame(rows)
    roster = {"HealthyC-H", "InjuredC-H", "OF1-H", "OF2-H"}

    critical = compute_critical_slots(roster, players)
    assert "HealthyC-H" in critical and "C" in critical["HealthyC-H"], (
        f"Sole startable catcher must be critical for C; got {critical}"
    )
    assert "InjuredC-H" not in critical, (
        "An IL player can't start, so dropping them never breaks the lineup — "
        f"they must not be protected. Got {critical}"
    )
    protected = find_protected_players(roster, players)
    assert "HealthyC-H" in protected


def test_validate_transaction_blocks_dropping_only_startable_catcher():
    """Dropping the only healthy catcher must fail slot feasibility."""
    from optimizer.swap_evaluator import validate_transaction

    # Roster with exactly enough startable coverage everywhere; the only
    # healthy catcher is HealthyC. A second catcher exists but is on IL.
    rows = []
    rows.append(
        {
            **_make_hitter(
                "HealthyC-H",
                pa=300,
                r=30,
                hr=8,
                rbi=30,
                sb=2,
                ops=0.700,
                owner="The Big Dumpers",
                position="C",
                roster_status="active",
            ),
            "injury_status": None,
        }
    )
    rows.append(
        {
            **_make_hitter(
                "InjuredC-H",
                pa=350,
                r=45,
                hr=15,
                rbi=45,
                sb=1,
                ops=0.800,
                owner="The Big Dumpers",
                position="C",
                roster_status="IR",
            ),
            "injury_status": "IL",
        }
    )
    for i, pos in enumerate(
        [
            "1B",
            "2B",
            "SS",
            "3B",
            "OF",
            "OF",
            "OF",
            "OF",
            "OF",
            "1B",
            "2B",
            "SS",
            "3B",
            "OF",
        ]
    ):
        rows.append(
            {
                **_make_hitter(
                    f"H{i}-H",
                    pa=450,
                    r=60,
                    hr=15,
                    rbi=60,
                    sb=8,
                    ops=0.760,
                    owner="The Big Dumpers",
                    position=pos,
                    roster_status="active",
                ),
                "injury_status": None,
            }
        )
    for i in range(8):
        rows.append(
            {
                **_make_pitcher(
                    f"SP{i}-P",
                    ip=120,
                    w=8,
                    sv=0,
                    k=120,
                    era=3.8,
                    whip=1.2,
                    owner="The Big Dumpers",
                    position="SP",
                    roster_status="active",
                ),
                "injury_status": None,
            }
        )
    for i in range(4):
        rows.append(
            {
                **_make_pitcher(
                    f"RP{i}-P",
                    ip=50,
                    w=3,
                    sv=10,
                    k=55,
                    era=3.2,
                    whip=1.1,
                    owner="The Big Dumpers",
                    position="RP",
                    roster_status="active",
                ),
                "injury_status": None,
            }
        )
    rows.append(
        {
            **_make_hitter(
                "FA_OF-H", pa=400, r=55, hr=18, rbi=55, sb=6, ops=0.790, position="OF"
            ),
            "injury_status": None,
        }
    )
    rows.append(
        {
            **_make_hitter(
                "FA_C-H", pa=380, r=40, hr=12, rbi=42, sb=3, ops=0.740, position="C"
            ),
            "injury_status": None,
        }
    )
    players = pd.DataFrame(rows)
    roster = set(players[players["owner"] == "The Big Dumpers"]["Name"])

    # Drop the only startable catcher for an OF: must be invalid.
    bad = validate_transaction({"HealthyC-H"}, {"FA_OF-H"}, roster, players)
    assert not bad["valid"], (
        "Dropping the only startable catcher for an OF must be rejected "
        f"(IL catcher can't fill C). Errors: {bad['errors']}"
    )
    assert any("C slot" in e for e in bad["errors"]), bad["errors"]

    # Drop the catcher FOR A BETTER CATCHER: must be valid.
    good = validate_transaction({"HealthyC-H"}, {"FA_C-H"}, roster, players)
    assert good["valid"], (
        f"Catcher-for-catcher swap must be allowed. Errors: {good['errors']}"
    )


# ============================================================================
# Monte Carlo standings simulation
# ============================================================================


def test_simulate_standings_consistent_with_ew():
    """E[my standing points] from simulation must equal 10 + EW (2-opp league: 20·? no — scaled)."""
    from optimizer.win_model import simulate_standings

    my_totals, opponent_totals = _synthetic_totals()
    sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    ew, _ = compute_win_probability(my_totals, opponent_totals, sigmas)

    sim = simulate_standings(my_totals, opponent_totals, sigmas, n_sims=60_000, seed=1)

    # With 2 opponents (3 teams) and 10 categories, expected points = 10 + EW
    # (points per category = 1 + #beaten). Monte Carlo noise ~ 0.05.
    expected = 10 + ew
    got = sim["expected_points"][0]
    assert abs(got - expected) < 0.15, (
        f"Simulated expected points {got:.2f} should match analytic "
        f"10 + EW = {expected:.2f}. The distributions are identical by "
        f"construction; a gap means a points/ranking bug."
    )
    # Probabilities are well-formed and sum to 1 across teams.
    assert abs(sum(sim["p_win"].values()) - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in sim["p_top2"].values())


# ============================================================================
# Lineup stability: untouched player-type half held fixed in swaps
# ============================================================================


def _full_roster_players():
    """A valid 28-man roster (16 H + 12 P) plus a couple of FAs, with MEW."""
    rows = []
    hpos = ["C", "1B", "2B", "SS", "3B", "OF", "OF", "OF", "OF", "OF",
            "1B", "2B", "SS", "3B", "OF", "C"]
    for i, pos in enumerate(hpos):
        rows.append({**_make_hitter(f"H{i}-H", pa=400 + i, r=55 + i, hr=12 + (i % 7),
                                    rbi=55 + i, sb=4 + (i % 9), ops=0.700 + 0.005 * (i % 6),
                                    owner="The Big Dumpers", position=pos,
                                    roster_status="active"), "injury_status": None})
    for i in range(8):
        rows.append({**_make_pitcher(f"SP{i}-P", ip=110 + 3 * i, w=7 + (i % 5), sv=0,
                                     k=110 + 4 * i, era=3.5 + 0.1 * (i % 6),
                                     whip=1.10 + 0.02 * (i % 5), owner="The Big Dumpers",
                                     position="SP", roster_status="active"),
                     "injury_status": None})
    for i in range(4):
        rows.append({**_make_pitcher(f"RP{i}-P", ip=45 + i, w=2 + i, sv=8 + 2 * i,
                                     k=50 + i, era=3.0 + 0.1 * i, whip=1.05 + 0.02 * i,
                                     owner="The Big Dumpers", position="RP",
                                     roster_status="active"), "injury_status": None})
    rows.append({**_make_hitter("FA_OF-H", pa=420, r=62, hr=20, rbi=60, sb=18, ops=0.760,
                                position="OF"), "injury_status": None})
    rows.append({**_make_pitcher("FA_SP-P", ip=140, w=11, sv=0, k=160, era=3.2,
                                 whip=1.05, position="SP"), "injury_status": None})
    players = pd.DataFrame(rows)
    my_totals, opp = _synthetic_totals()
    sig = estimate_projection_uncertainty(my_totals, opp)
    grad = compute_ew_gradient(my_totals, opp, sig)
    players = add_mew(players, my_totals, grad)
    return players, opp, sig


def test_hitter_swap_leaves_pitching_totals_unchanged():
    """A hitter-for-hitter swap must not change any pitching total."""
    from optimizer.lineup_solver import compute_totals_for_starters
    from optimizer.swap_evaluator import compute_exact_msv
    from optimizer.win_model import compute_win_probability

    players, opp, sig = _full_roster_players()
    roster = set(players[players["owner"] == "The Big Dumpers"]["Name"])
    base_lineup = solve_lineup(roster, players, "MEW")
    base_totals = compute_totals_for_starters(set(base_lineup.keys()), players)
    cur_ew, _ = compute_win_probability(base_totals, opp, sig)

    res = compute_exact_msv({"H5-H"}, {"FA_OF-H"}, roster, players, opp, sig,
                            cur_ew, baseline_lineup=base_lineup)
    for cat in ("W", "SV", "K", "ERA", "WHIP"):
        assert abs(res["new_totals"][cat] - base_totals[cat]) < 1e-9, (
            f"Hitter swap changed pitching {cat}: "
            f"{base_totals[cat]} → {res['new_totals'][cat]}"
        )


def test_solve_lineup_half_is_separable():
    """Solving only HITTING_SLOTS yields exactly the hitter half of a full solve."""
    from optimizer.config import HITTING_SLOTS

    players, _, _ = _full_roster_players()
    roster = set(players[players["owner"] == "The Big Dumpers"]["Name"])
    full = solve_lineup(roster, players, "MEW")
    full_h = {n: s for n, s in full.items() if s in HITTING_SLOTS}
    half = solve_lineup(roster, players, "MEW", slots=HITTING_SLOTS)
    assert set(half.keys()) == set(full_h.keys()), (
        f"Hitter-half solve differs from full solve's hitter half:\n"
        f"  half:  {sorted(half.keys())}\n  full:  {sorted(full_h.keys())}"
    )


def test_validate_balanced_swap_no_size_warning():
    """A 1-for-1 swap on a roster that includes an IR player must not warn on size."""
    from optimizer.swap_evaluator import validate_transaction

    players, _, _ = _full_roster_players()
    # Put one player on IR so the working roster exceeds the active-28 constant.
    players.loc[players["Name"] == "H15-H", "roster_status"] = "IR"
    roster = set(players[players["owner"] == "The Big Dumpers"]["Name"])
    res = validate_transaction({"H5-H"}, {"FA_OF-H"}, roster, players)
    assert not any("size changes" in w for w in res["warnings"]), (
        f"Balanced 1-for-1 swap should not trigger a roster-size warning; "
        f"got {res['warnings']}"
    )
