"""
League state computation with MEW-lineup fixed-point iteration.

Central state-computation step from MATHEMATICAL_FRAMEWORK §8.
Depends on config, players, lineup_solver, player_scoring, win_model.
"""

import pandas as pd

from .config import MY_TEAM_NAME, N_STARTER_SLOTS
from .lineup_solver import (
    compute_totals_for_starters,
    maybe_blend,
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


def _complete_banked_volume(
    banked: dict[str, float] | None,
    ros: dict[str, float],
    fraction_remaining: float,
) -> dict[str, float] | None:
    """Fill a banked totals dict with estimated PA/IP weights for ratio blending.

    The standings source provides banked category values but not banked PA/IP.
    Under roughly uniform play, elapsed volume relates to remaining volume by
    (1 − f)/f, so banked_PA ≈ ros_PA · (1 − f)/f (and likewise IP). These are
    only the *weights* for blending banked vs. ros OPS/ERA/WHIP, so the
    uniform-play approximation is adequate. Banked counting/rate values come
    straight from the authoritative standings.

    Returns a new dict (banked is not mutated); None passes through unchanged.
    """
    if banked is None:
        return None
    out = dict(banked)
    ratio = (
        (1.0 - fraction_remaining) / fraction_remaining
        if fraction_remaining < 1.0
        else 0.0
    )
    if out.get("PA", 0.0) <= 0.0:
        out["PA"] = float(ros["PA"]) * ratio
    if out.get("IP", 0.0) <= 0.0:
        out["IP"] = float(ros["IP"]) * ratio
    return out


def compute_league_state(
    players: pd.DataFrame,
    my_team_name: str | None = None,
    banked_totals: dict[str, dict[str, float]] | None = None,
    season_fraction_remaining: float | None = None,
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
        banked_totals: Optional banked YTD totals keyed by team NAME. Each value
            is a totals dict (10 categories + 'PA' + 'IP') in the
            compute_totals_for_starters shape. When provided, every team's
            standings totals become banked + rest-of-season (blend_season_totals);
            when None, the legacy rest-of-season-only behavior is used. Teams
            absent from the dict fall back to ros-only for that team.
        season_fraction_remaining: Fraction of the season still to be played,
            in (0, 1]. Used to rescale σ to the remaining horizon (σ ∝ √f).
            Only applied when banked_totals is provided (the σ scaling assumes a
            banked+ros season model). None disables the rescaling.

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

    # Banked modeling requires a season fraction to estimate PA/IP blend weights
    # and to rescale σ to the remaining horizon.
    if banked_totals is not None:
        assert (
            season_fraction_remaining is not None
            and 0.0 < season_fraction_remaining <= 1.0
        ), (
            "compute_league_state: banked_totals requires season_fraction_remaining "
            f"in (0, 1]; got {season_fraction_remaining}."
        )

    # Only apply the σ horizon rescaling when modeling banked+ros seasons.
    sigma_fraction = season_fraction_remaining if banked_totals is not None else None
    my_banked_raw = (
        banked_totals.get(my_team_name) if banked_totals is not None else None
    )

    # --- 1. Identify rosters ---
    opponent_teams = sorted(
        t for t in work[work["owner"].notna()]["owner"].unique() if t != my_team_name
    )
    assert len(opponent_teams) > 0, "compute_league_state: no opponent teams found"
    my_roster_names = get_main_roster(work, my_team_name)

    # Banked modeling must be all-or-nothing across teams: comparing one team's
    # banked+ros season total against another's ros-only total is meaningless.
    # If any participating team is absent from banked_totals (e.g. a standings
    # team-name mismatch), fall back to rest-of-season-only for everyone.
    if banked_totals is not None:
        participating = {my_team_name, *opponent_teams}
        missing_banked = participating - set(banked_totals)
        if missing_banked:
            print(
                f"WARNING: banked totals missing for {sorted(missing_banked)} "
                f"(team-name mismatch with standings?). Falling back to "
                f"rest-of-season-only for ALL teams to keep comparisons valid."
            )
            banked_totals = None
            sigma_fraction = None
            my_banked_raw = None

    # --- 2. Opponent lineups (FV) ---
    # ros totals feed σ (only the unknown half is uncertain); season totals
    # (banked + ros) feed the gradient and EW.
    opponent_lineups: dict[int, dict[str, str]] = {}
    opponent_rosters: dict[int, set[str]] = {}
    opponent_ros_totals: dict[int, dict[str, float]] = {}
    opponent_totals: dict[int, dict[str, float]] = {}
    opponent_banked: dict[int, dict[str, float] | None] = {}

    for i, team in enumerate(opponent_teams):
        opp_id = i + 1
        opp_roster = get_main_roster(work, team)
        opponent_rosters[opp_id] = opp_roster
        opp_lineup = solve_lineup(opp_roster, work, "FV")
        opponent_lineups[opp_id] = opp_lineup
        opp_ros = compute_totals_for_starters(set(opp_lineup.keys()), work)
        opp_banked_raw = banked_totals.get(team) if banked_totals is not None else None
        opp_banked = (
            _complete_banked_volume(opp_banked_raw, opp_ros, season_fraction_remaining)
            if opp_banked_raw is not None
            else None
        )
        opponent_banked[opp_id] = opp_banked
        opponent_ros_totals[opp_id] = opp_ros
        opponent_totals[opp_id] = maybe_blend(opp_banked, opp_ros)

    print(f"Opponent lineups solved (FV) for {len(opponent_teams)} teams")

    # --- 3. My team: initial solve with FV ---
    my_lineup = solve_lineup(my_roster_names, work, "FV")
    my_starters = set(my_lineup.keys())
    my_ros = compute_totals_for_starters(my_starters, work)
    # Fix the banked PA/IP weights once from the initial ros volume; banked
    # volume is a property of the past and does not change with future roster.
    my_banked = (
        _complete_banked_volume(my_banked_raw, my_ros, season_fraction_remaining)
        if my_banked_raw is not None
        else None
    )

    assert len(my_starters) == N_STARTER_SLOTS, (
        f"compute_league_state: my lineup has {len(my_starters)} starters, "
        f"expected {N_STARTER_SLOTS}"
    )

    # --- 4. MEW-lineup iteration with best-EW selection ---
    # Each candidate lineup gets its full state evaluated (totals, σ, gradient,
    # exact EW — all closed-form, no extra MILP). The iteration follows the
    # MEW best-response map; visiting a previously-seen starter set means a
    # fixed point or a cycle, and in EITHER case the right answer is the
    # max-EW lineup among those visited. EW is the actual objective; Σ MEW is
    # only its linearization, so the fixed point need not be EW-optimal.
    def _eval_lineup(lineup: dict[str, str]) -> dict:
        starters = set(lineup.keys())
        ros = compute_totals_for_starters(starters, work)
        totals = maybe_blend(my_banked, ros)
        sig = estimate_projection_uncertainty(ros, opponent_ros_totals, sigma_fraction)
        grad = compute_ew_gradient(totals, opponent_totals, sig)
        ew, _ = compute_win_probability(totals, opponent_totals, sig)
        return {
            "lineup": lineup,
            "starters": starters,
            "ros": ros,
            "totals": totals,
            "sigmas": sig,
            "gradient": grad,
            "ew": ew,
        }

    visited: dict[frozenset, dict] = {}
    cur = _eval_lineup(my_lineup)
    visited[frozenset(cur["starters"])] = cur

    for iteration in range(_MAX_LINEUP_ITERATIONS):
        work = add_mew(work, cur["totals"], cur["gradient"])
        new_lineup = solve_lineup(my_roster_names, work, "MEW")
        key = frozenset(new_lineup.keys())
        if key in visited:
            if visited[key] is cur:
                print(f"MEW-lineup converged after {iteration + 1} iteration(s)")
            else:
                print(
                    f"MEW-lineup cycling over {len(visited)} lineups after "
                    f"{iteration + 1} iteration(s); selecting max-EW lineup."
                )
            break
        cur = _eval_lineup(new_lineup)
        visited[key] = cur
    else:
        print(
            f"MEW-lineup still moving after {_MAX_LINEUP_ITERATIONS} "
            f"iterations; selecting max-EW lineup of {len(visited)} visited."
        )

    best = max(visited.values(), key=lambda s: s["ew"])

    # Export a SELF-CONSISTENT snapshot: the exported lineup must equal a fresh
    # MEW solve under the exported gradient, or every downstream identity swap
    # (re-solving the lineup) drifts and injects phantom category deltas. We
    # anchor on the best-EW gradient and hold it FIXED while letting the lineup
    # settle — with the gradient fixed there is no oscillation, only the weak
    # ratio-baseline feedback, which converges in 1–2 steps.
    gradient = best["gradient"]
    sigmas = best["sigmas"]
    my_totals = best["totals"]
    my_lineup = best["lineup"]
    for _ in range(_MAX_LINEUP_ITERATIONS):
        work = add_mew(work, my_totals, gradient)
        settled = solve_lineup(my_roster_names, work, "MEW")
        if set(settled.keys()) == set(my_lineup.keys()):
            my_lineup = settled
            break
        my_lineup = settled
        my_totals = maybe_blend(
            my_banked, compute_totals_for_starters(set(my_lineup.keys()), work)
        )
    my_starters = set(my_lineup.keys())
    my_ros = compute_totals_for_starters(my_starters, work)
    my_totals = maybe_blend(my_banked, my_ros)
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
        "my_banked": my_banked,
        "opponent_banked": opponent_banked,
        "season_fraction_remaining": sigma_fraction,
    }
