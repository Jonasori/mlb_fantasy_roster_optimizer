"""
Lineup slot assignment (MILP) and team total aggregation.

Given a roster, who starts where and what are the totals?
"""

from collections.abc import Iterable

import pandas as pd
import pulp
from pulp import LpVariable, lpSum

from .config import (
    HITTING_SLOTS,
    PITCHING_SLOTS,
)
from .players import get_startable_slots

# ============================================================================
# LINEUP ASSIGNMENT (MILP)
# ============================================================================

_ALL_SLOTS: dict[str, int] = {**HITTING_SLOTS, **PITCHING_SLOTS}


def get_solver() -> pulp.LpSolver:
    """Return the MILP solver for lineup/roster problems.

    Uses HiGHS (via highspy) per the framework spec — it is fast (<1ms for
    these tiny assignment problems) and ships as a portable wheel, unlike
    PuLP's bundled CBC binary (which is x86-only and fails on arm64 Macs).
    """
    return pulp.HiGHS(msg=False)


def solve_lineup(
    roster_names: Iterable[str],
    players: pd.DataFrame,
    objective_column: str = "FV",
    force_start: set[str] | None = None,
    slots: dict[str, int] | None = None,
) -> dict[str, str]:
    """Solve lineup assignment via MILP, maximizing Σ objective_column for starters.

    The objective_column parameter determines what the lineup optimizes:
    - "FV" for opponents (context-free z-score sum, no gradient needed)
    - "MEW" for my team (gradient-weighted, context-aware)

    My team uses MEW because the gradient g_c weights each category by
    its current marginal value — the lineup prioritizes players who
    contribute most to categories where improvement matters. Opponents
    use FV because we model them as optimizing context-free quality,
    not optimizing against specific category needs.

    With 28 binary variables and ~18 slot constraints, solves in <1ms.

    Args:
        roster_names: Player names on the roster (with -H/-P suffix).
        players: Players DataFrame with the objective_column.
        objective_column: Column to maximize over starters.
        force_start: Optional set of player names that must be assigned
            to a starter slot. Used for "what-if" lineup comparisons.
        slots: Optional slot subset to fill (e.g. HITTING_SLOTS to solve only
            the hitter half). Defaults to all slots. The hitter and pitcher
            halves are independent (disjoint players and slots), so solving one
            half is exact — used to hold the untouched half of a roster fixed
            during single-type swap evaluation.

    Returns:
        Dict mapping starter name → assigned slot. Bench players omitted.
    """
    active_slots = slots if slots is not None else _ALL_SLOTS
    roster_names = set(roster_names)
    roster_df = players[players["Name"].isin(roster_names)].copy()

    found_names = set(roster_df["Name"])
    missing = roster_names - found_names
    assert len(missing) == 0, (
        f"solve_lineup: {len(missing)} players not found in DataFrame: "
        f"{sorted(missing)}"
    )
    assert objective_column in roster_df.columns, (
        f"solve_lineup: objective column '{objective_column}' not in DataFrame. "
        f"Available columns: {sorted(roster_df.columns)}"
    )

    if force_start:
        missing_forced = force_start - roster_names
        assert len(missing_forced) == 0, (
            f"solve_lineup: force_start players not on roster: {sorted(missing_forced)}"
        )

    roster_df = roster_df.reset_index(drop=True)
    n_players = len(roster_df)
    name_to_idx = {roster_df.iloc[i]["Name"]: i for i in range(n_players)}

    has_injury_col = "injury_status" in roster_df.columns
    eligibility = {
        i: get_startable_slots(
            roster_df.iloc[i]["Position"],
            roster_df.iloc[i]["injury_status"] if has_injury_col else None,
        )
        & set(active_slots)
        for i in range(n_players)
    }

    prob = pulp.LpProblem("LineupAssignment", pulp.LpMaximize)

    a: dict[tuple[int, str], LpVariable] = {}
    for i in range(n_players):
        for s in eligibility[i]:
            a[i, s] = LpVariable(f"a_{i}_{s}", cat="Binary")

    prob += lpSum(
        roster_df.iloc[i][objective_column]
        * lpSum(a[i, s] for s in eligibility[i] if (i, s) in a)
        for i in range(n_players)
    )

    unfillable: list[str] = []
    for slot, count in active_slots.items():
        eligible_for_slot = [i for i in range(n_players) if slot in eligibility[i]]
        slots_to_fill = min(len(eligible_for_slot), count)
        if slots_to_fill < count:
            unfillable.append(f"{slot} ({slots_to_fill}/{count})")
        if slots_to_fill > 0:
            prob += (
                lpSum(a[i, slot] for i in eligible_for_slot if (i, slot) in a)
                == slots_to_fill,
                f"fill_{slot}",
            )

    if unfillable:
        # Leniency keeps opponent solves robust (e.g. all their catchers on
        # IL), but an under-filled lineup forfeits starter stats — screening
        # and validate_transaction should have prevented this for MY moves.
        print(
            f"WARNING: solve_lineup leaving slots under-filled: "
            f"{', '.join(unfillable)}. Roster lacks startable coverage."
        )

    for i in range(n_players):
        slots_for_player = [s for s in eligibility[i] if (i, s) in a]
        if slots_for_player:
            prob += lpSum(a[i, s] for s in slots_for_player) <= 1, f"one_slot_{i}"

    if force_start:
        for name in force_start:
            i = name_to_idx[name]
            slots_for_player = [s for s in eligibility[i] if (i, s) in a]
            assert len(slots_for_player) > 0, (
                f"solve_lineup: force_start player '{name}' has no eligible slots"
            )
            prob += (
                lpSum(a[i, s] for s in slots_for_player) == 1,
                f"force_{i}",
            )

    prob.solve(get_solver())

    assert prob.status == pulp.LpStatusOptimal, (
        f"Lineup assignment failed: {pulp.LpStatus[prob.status]}. "
        f"Roster has {n_players} players. Check position eligibility."
    )

    slot_assignments: dict[str, str] = {}
    for i in range(n_players):
        for s in eligibility[i]:
            if (i, s) in a and pulp.value(a[i, s]) > 0.5:
                slot_assignments[roster_df.iloc[i]["Name"]] = s
                break

    return slot_assignments


# ============================================================================
# TEAM TOTAL COMPUTATION
# ============================================================================


def compute_team_totals(
    roster_names: Iterable[str],
    players: pd.DataFrame,
    objective_column: str = "FV",
) -> dict[str, float]:
    """Solve lineup MILP then aggregate starters into team totals.

    This is the workhorse function: MILP + aggregation in one call.
    Passes objective_column through to solve_lineup.

    CRITICAL: ratio stats are PA/IP-weighted averages, NOT sums.
        OPS = Σ(PA × OPS) / Σ(PA)
        ERA = Σ(IP × ERA) / Σ(IP)
        WHIP = Σ(IP × WHIP) / Σ(IP)

    Args:
        roster_names: Player names on the roster (with -H/-P suffix).
        players: Players DataFrame.
        objective_column: Passed to solve_lineup.

    Returns:
        Dict mapping category to team total, plus 'PA' and 'IP'.
    """
    lineup = solve_lineup(roster_names, players, objective_column)
    return compute_totals_for_starters(set(lineup.keys()), players)


def compute_totals_for_starters(
    starters: set[str],
    players: pd.DataFrame,
) -> dict[str, float]:
    """Team totals for a known set of starters (no MILP). Fast path.

    Returns dict with keys for all 10 categories PLUS 'PA' and 'IP':
        {'R': 823, 'HR': 245, ..., 'ERA': 3.85, ..., 'PA': 5200, 'IP': 1100}

    PA and IP are simple sums of starters. These are needed by the MEW
    formula (ratio stat baselines and total-weight denominators).
    Counting stats are sums; ratio stats are weighted averages.
    """
    team_df = players[players["Name"].isin(starters)]

    found = set(team_df["Name"])
    missing = starters - found
    assert len(missing) == 0, (
        f"compute_totals_for_starters: {len(missing)} starters not in DataFrame: "
        f"{sorted(missing)}"
    )

    hitters = team_df[team_df["player_type"] == "hitter"]
    pitchers = team_df[team_df["player_type"] == "pitcher"]

    totals: dict[str, float] = {}

    for cat in ("R", "HR", "RBI", "SB"):
        totals[cat] = float(hitters[cat].sum())

    total_pa = float(hitters["PA"].sum())
    assert total_pa > 0, (
        f"Total PA is 0 — no hitters with plate appearances in starters. "
        f"Starters: {sorted(starters)}"
    )
    totals["OPS"] = float((hitters["PA"] * hitters["OPS"]).sum() / total_pa)
    totals["PA"] = total_pa

    for cat in ("W", "SV", "K"):
        totals[cat] = float(pitchers[cat].sum())

    total_ip = float(pitchers["IP"].sum())
    assert total_ip > 0, (
        f"Total IP is 0 — no pitchers with innings in starters. "
        f"Starters: {sorted(starters)}"
    )
    totals["ERA"] = float((pitchers["IP"] * pitchers["ERA"]).sum() / total_ip)
    totals["WHIP"] = float((pitchers["IP"] * pitchers["WHIP"]).sum() / total_ip)
    totals["IP"] = total_ip

    return totals


# ============================================================================
# BANKED-YTD / REST-OF-SEASON BLENDING
# ============================================================================

_COUNTING_TOTAL_KEYS: tuple[str, ...] = ("R", "HR", "RBI", "SB", "W", "SV", "K")


def blend_season_totals(
    banked: dict[str, float],
    ros: dict[str, float],
) -> dict[str, float]:
    """Combine banked year-to-date totals with rest-of-season projected totals.

    Roto standings are decided by FULL-SEASON totals = what a team has already
    banked (YTD) plus what it will accrue the rest of the way (ROS). The
    projection feeds are rest-of-season only, so the win model must add the
    banked half before computing matchup z-scores. This is the single most
    important correction once projections are RoS: without it, the model
    treats every category race as starting from a tie.

    Counting stats are summed. Ratio stats are re-weighted by the playing-time
    denominator, matching the team-total convention in
    ``compute_totals_for_starters`` (PA-weighted OPS, IP-weighted ERA/WHIP):

        season_OPS  = (PA_b·OPS_b  + PA_r·OPS_r)  / (PA_b + PA_r)
        season_ERA  = (IP_b·ERA_b  + IP_r·ERA_r)  / (IP_b + IP_r)
        season_WHIP = (IP_b·WHIP_b + IP_r·WHIP_r) / (IP_b + IP_r)

    Args:
        banked: Banked YTD totals. Must contain the 10 category keys plus
            'PA' and 'IP' (same shape as ``compute_totals_for_starters``).
            PA/IP are the banked playing-time weights for ratio blending.
        ros: Rest-of-season projected totals (same shape).

    Returns:
        Season totals dict with the same keys (10 categories + 'PA' + 'IP').
    """
    required = set(_COUNTING_TOTAL_KEYS) | {"OPS", "ERA", "WHIP", "PA", "IP"}
    missing_b = required - set(banked)
    missing_r = required - set(ros)
    assert not missing_b, (
        f"blend_season_totals: banked totals missing keys {sorted(missing_b)}. "
        f"Banked dict must match compute_totals_for_starters shape."
    )
    assert not missing_r, (
        f"blend_season_totals: ros totals missing keys {sorted(missing_r)}."
    )

    season: dict[str, float] = {}
    for cat in _COUNTING_TOTAL_KEYS:
        season[cat] = float(banked[cat]) + float(ros[cat])

    pa_b, pa_r = float(banked["PA"]), float(ros["PA"])
    ip_b, ip_r = float(banked["IP"]), float(ros["IP"])
    season["PA"] = pa_b + pa_r
    season["IP"] = ip_b + ip_r

    season["OPS"] = (
        (pa_b * float(banked["OPS"]) + pa_r * float(ros["OPS"])) / (pa_b + pa_r)
        if (pa_b + pa_r) > 0
        else float(ros["OPS"])
    )
    season["ERA"] = (
        (ip_b * float(banked["ERA"]) + ip_r * float(ros["ERA"])) / (ip_b + ip_r)
        if (ip_b + ip_r) > 0
        else float(ros["ERA"])
    )
    season["WHIP"] = (
        (ip_b * float(banked["WHIP"]) + ip_r * float(ros["WHIP"])) / (ip_b + ip_r)
        if (ip_b + ip_r) > 0
        else float(ros["WHIP"])
    )
    return season


def maybe_blend(
    banked: dict[str, float] | None,
    ros: dict[str, float],
) -> dict[str, float]:
    """Blend banked + ros if banked is provided; otherwise return ros unchanged.

    Lets every evaluator accept an optional ``banked`` baseline without
    branching: ``maybe_blend(None, ros)`` is the pure rest-of-season path
    (backward-compatible), ``maybe_blend(banked, ros)`` is the season path.
    """
    return blend_season_totals(banked, ros) if banked is not None else ros


# ============================================================================
# ASSIGN OPTIMAL SLOTS TO DATAFRAME
# ============================================================================


def assign_optimal_slots(
    players: pd.DataFrame,
    my_lineup: dict[str, str],
    opponent_lineups: dict[int, dict[str, str]],
    opponent_teams: list[str],
) -> pd.DataFrame:
    """Set optimal_slot column from pre-computed lineup assignments.

    Takes lineups produced by compute_league_state (which runs the
    MEW-lineup fixed-point iteration for my team, FV for opponents).

    Args:
        players: The players DataFrame.
        my_lineup: {player_name: slot} for my team's MEW-optimal lineup.
        opponent_lineups: {opponent_id: {player_name: slot}} for each opponent.
        opponent_teams: Sorted list of opponent team names (maps to opponent IDs).

    Players not in any lineup get optimal_slot = None.
    """
    players = players.copy()
    players["optimal_slot"] = None

    for name, slot in my_lineup.items():
        mask = players["Name"] == name
        assert mask.any(), (
            f"assign_optimal_slots: my lineup player '{name}' not in DataFrame"
        )
        players.loc[mask, "optimal_slot"] = slot

    for opp_id, lineup in opponent_lineups.items():
        for name, slot in lineup.items():
            mask = players["Name"] == name
            assert mask.any(), (
                f"assign_optimal_slots: opponent {opp_id} player '{name}' "
                f"not in DataFrame"
            )
            players.loc[mask, "optimal_slot"] = slot

    return players
