"""
League state computation with MEW-lineup fixed-point iteration.

Central state-computation step from MATHEMATICAL_FRAMEWORK §8.
Depends on config, players, lineup_solver, player_scoring, win_model.
"""

import pandas as pd

from .config import MY_TEAM_NAME, N_STARTER_SLOTS
from .lineup_solver import (
    compute_totals_for_starters,
    solve_lineup,
)
from .player_scoring import add_mew
from .rosters import get_main_roster
from .win_model import (
    compute_ew_gradient,
    compute_win_probability,
    estimate_projection_uncertainty,
)

_MAX_LINEUP_ITERATIONS: int = 5


def compute_league_state(
    players: pd.DataFrame, my_team_name: str | None = None
) -> dict:
    """Compute converged league state via MEW-lineup fixed-point iteration.

    The state computation step from MATHEMATICAL_FRAMEWORK §8. Produces
    everything needed for player scoring, screening, and evaluation.

    Algorithm:
        1. Opponent lineups: solve each opponent's lineup with FV
           (one MILP each). Compute opponent_totals.

        2. My team: MEW-lineup fixed-point iteration (MATH_FRAMEWORK §5)
           a. Solve initial lineup with FV → my_starters₀, my_totals₀
           b. estimate_projection_uncertainty → category_sigmas
           c. compute_ew_gradient → gradient
           d. Compute MEW for all players via add_mew (on working copy)
           e. Re-solve my lineup with objective_column="MEW" → my_starters₁
           f. If my_starters₁ ≠ my_starters₀: update totals, go to (b)
           g. Converged when starter set stabilizes

        3. Compute current_ew from converged state

    Convergence: improving in category c increases z_{c,o}, decreasing
    φ(z_{c,o}), decreasing |g_c|. Concavity of Φ acts as a damper.
    Worst case: 2 possible lineups — evaluate both, pick higher EW.

    Args:
        players: DataFrame with FV column (from add_fantasy_value).
        my_team_name: Which team to treat as "my team". Defaults to
            MY_TEAM_NAME from config if not provided.

    Opponent ID convention (from MATHEMATICAL_FRAMEWORK §1):
        Opponent IDs are 1-indexed: O = {1, ..., 6}.
        opponent_teams is sorted alphabetically; the i-th name (1-indexed)
        maps to opponent ID i.

    Returns:
        {
            'my_totals': dict[str, float],
            'opponent_totals': dict[int, dict[str, float]],
            'category_sigmas': dict[str, float],
            'gradient': dict[str, float],
            'my_roster_names': set[str],
            'opponent_rosters': dict[int, set[str]],
            'my_starters': set[str],
            'my_lineup': dict[str, str],
            'opponent_lineups': dict[int, dict[str, str]],
            'opponent_teams': list[str],
            'current_ew': float,
            'my_team_name': str,
        }
    """
    if my_team_name is None:
        my_team_name = MY_TEAM_NAME
    assert "FV" in players.columns, (
        "compute_league_state: players must have FV column. "
        "Call add_fantasy_value() first."
    )

    work = players.copy()

    # --- 1. Identify rosters ---
    opponent_teams = sorted(
        t for t in work[work["owner"].notna()]["owner"].unique() if t != my_team_name
    )
    assert len(opponent_teams) > 0, "compute_league_state: no opponent teams found"
    my_roster_names = get_main_roster(work, my_team_name)

    # --- 2. Opponent lineups (FV) ---
    opponent_lineups: dict[int, dict[str, str]] = {}
    opponent_rosters: dict[int, set[str]] = {}
    opponent_totals: dict[int, dict[str, float]] = {}

    for i, team in enumerate(opponent_teams):
        opp_id = i + 1
        opp_roster = get_main_roster(work, team)
        opponent_rosters[opp_id] = opp_roster
        opp_lineup = solve_lineup(opp_roster, work, "FV")
        opponent_lineups[opp_id] = opp_lineup
        opponent_totals[opp_id] = compute_totals_for_starters(
            set(opp_lineup.keys()), work
        )

    print(f"Opponent lineups solved (FV) for {len(opponent_teams)} teams")

    # --- 3. My team: initial solve with FV ---
    my_lineup = solve_lineup(my_roster_names, work, "FV")
    my_starters = set(my_lineup.keys())
    my_totals = compute_totals_for_starters(my_starters, work)

    assert len(my_starters) == N_STARTER_SLOTS, (
        f"compute_league_state: my lineup has {len(my_starters)} starters, "
        f"expected {N_STARTER_SLOTS}"
    )

    # --- 4. MEW-lineup fixed-point iteration ---
    sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)
    work = add_mew(work, my_totals, gradient)

    prev_starters = my_starters
    converged = False

    for iteration in range(_MAX_LINEUP_ITERATIONS):
        new_lineup = solve_lineup(my_roster_names, work, "MEW")
        new_starters = set(new_lineup.keys())

        if new_starters == prev_starters:
            my_lineup = new_lineup
            converged = True
            print(f"MEW-lineup converged after {iteration + 1} iteration(s)")
            break

        my_lineup = new_lineup
        my_starters = new_starters
        my_totals = compute_totals_for_starters(my_starters, work)
        sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
        gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)
        work = add_mew(work, my_totals, gradient)
        prev_starters = new_starters

    if not converged:
        # Oscillation between two lineups: evaluate both, pick higher EW
        print(
            f"WARNING: MEW-lineup did not converge after "
            f"{_MAX_LINEUP_ITERATIONS} iterations. "
            f"Evaluating final two lineups to pick the better one."
        )
        lineup_a = my_lineup
        starters_a = set(lineup_a.keys())
        totals_a = compute_totals_for_starters(starters_a, work)
        ew_a, _ = compute_win_probability(totals_a, opponent_totals, sigmas)

        lineup_b = solve_lineup(my_roster_names, work, "MEW")
        starters_b = set(lineup_b.keys())
        totals_b = compute_totals_for_starters(starters_b, work)
        ew_b, _ = compute_win_probability(totals_b, opponent_totals, sigmas)

        if ew_b > ew_a:
            my_lineup = lineup_b
            my_starters = starters_b
            my_totals = totals_b
        else:
            my_lineup = lineup_a
            my_starters = starters_a
            my_totals = totals_a

        sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
        gradient = compute_ew_gradient(my_totals, opponent_totals, sigmas)

    # --- 5. Compute EW from converged state ---
    current_ew, _ = compute_win_probability(my_totals, opponent_totals, sigmas)

    print(
        f"League state: EW = {current_ew:.2f}/60, "
        f"standing points ≈ {10 + current_ew:.1f}"
    )

    return {
        "my_totals": my_totals,
        "opponent_totals": opponent_totals,
        "category_sigmas": sigmas,
        "gradient": gradient,
        "my_roster_names": my_roster_names,
        "opponent_rosters": opponent_rosters,
        "my_starters": my_starters,
        "my_lineup": my_lineup,
        "opponent_lineups": opponent_lineups,
        "opponent_teams": opponent_teams,
        "current_ew": current_ew,
        "my_team_name": my_team_name,
    }
