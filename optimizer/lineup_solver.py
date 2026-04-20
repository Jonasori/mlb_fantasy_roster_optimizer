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
from .players import get_eligible_slots

# ============================================================================
# LINEUP ASSIGNMENT (MILP)
# ============================================================================

_ALL_SLOTS: dict[str, int] = {**HITTING_SLOTS, **PITCHING_SLOTS}


def solve_lineup(
    roster_names: Iterable[str],
    players: pd.DataFrame,
    objective_column: str = "FV",
    force_start: set[str] | None = None,
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

    Returns:
        Dict mapping starter name → assigned slot. Bench players omitted.
    """
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

    eligibility = {
        i: get_eligible_slots(roster_df.iloc[i]["Position"]) for i in range(n_players)
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

    for slot, count in _ALL_SLOTS.items():
        eligible_for_slot = [i for i in range(n_players) if slot in eligibility[i]]
        slots_to_fill = min(len(eligible_for_slot), count)
        if slots_to_fill > 0:
            prob += (
                lpSum(a[i, slot] for i in eligible_for_slot if (i, slot) in a)
                == slots_to_fill,
                f"fill_{slot}",
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

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

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
