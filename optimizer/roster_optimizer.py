"""
Free Agent Optimizer using Mixed-Integer Linear Programming (MILP).

Answers: "Given a pool of available players, what is the optimal 26-player roster
that maximizes expected wins against known opponents?"
"""

import time

import numpy as np
import pandas as pd
import pulp
from pulp import LpVariable, lpSum, value
from tqdm.auto import tqdm

from .data_loader import (
    ALL_CATEGORIES,
    BALANCE_LAMBDA_DEFAULT,
    HITTING_CATEGORIES,
    HITTING_SLOTS,
    MAX_HITTERS,
    MAX_PITCHERS,
    MIN_HITTERS,
    MIN_PITCHERS,
    NEGATIVE_CATEGORIES,
    NUM_OPPONENTS,
    PITCHING_CATEGORIES,
    PITCHING_SLOTS,
    RATIO_STATS,
    ROSTER_SIZE,
    SLOT_ELIGIBILITY,
    compute_team_totals,
    estimate_projection_uncertainty,
    strip_name_suffix,
)

# =============================================================================
# MILP CONSTANTS
# =============================================================================

# Big-M values for indicator constraints
BIG_M_COUNTING = 10000  # For counting stats (R, HR, RBI, SB, W, SV, K)
BIG_M_RATIO = 5000  # For ratio stat linearized forms (OPS, ERA, WHIP)

# Epsilon values for strict inequality
EPSILON_COUNTING = 0.5  # For integer-valued counting stats
EPSILON_RATIO = 0.001  # For continuous ratio stats


# =============================================================================
# SLOT ELIGIBILITY
# =============================================================================


def compute_slot_eligibility(candidates: pd.DataFrame) -> dict[int, set[str]]:
    """
    Precompute which slots each candidate can fill.

    This handles multi-position players correctly:
        - "SS,2B" player can fill: SS, 2B, UTIL
        - "SP,RP" pitcher can fill: SP, RP
        - "OF" player can fill: OF, UTIL

    Args:
        candidates: Filtered candidates DataFrame with Position column

    Returns:
        Dict mapping candidate index to set of eligible slot types.
        Example: {0: {'SS', '2B', 'UTIL'}, 1: {'OF', 'UTIL'}, 2: {'SP', 'RP'}, ...}
    """
    eligibility = {}

    for i, row in candidates.iterrows():
        position_str = row["Position"]
        player_positions = set(p.strip() for p in str(position_str).split(","))

        eligible_slots = set()
        for slot, valid_positions in SLOT_ELIGIBILITY.items():
            if player_positions & valid_positions:  # Set intersection
                eligible_slots.add(slot)

        eligibility[i] = eligible_slots

    return eligibility


def validate_slot_coverage(
    eligibility: dict[int, set[str]],
    candidates: pd.DataFrame,
) -> None:
    """
    Verify enough candidates exist for each slot BEFORE solving.

    This catches infeasibility early with a clear error message,
    rather than getting an opaque "infeasible" from the solver.
    """
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}

    for slot, required_count in all_slots.items():
        candidates_for_slot = sum(1 for i in eligibility if slot in eligibility[i])

        assert candidates_for_slot >= required_count, (
            f"Cannot fill {slot} slot: need {required_count}, "
            f"but only {candidates_for_slot} candidates are eligible. "
            f"Increase top_n_per_position or add more candidates at {slot}."
        )


# =============================================================================
# CANDIDATE FILTERING
# =============================================================================


def filter_candidates(
    projections: pd.DataFrame,
    quality_scores: pd.DataFrame,
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    top_n_per_position: int = 30,
    top_n_per_category: int = 10,
) -> pd.DataFrame:
    """
    Filter to candidate players for optimization.

    Include:
        1. ALL players currently on my roster (must be candidates)
        2. Top N available players at each position by quality score
        3. Top M available players in each scoring category (specialists)

    Exclude:
        - All players on opponent rosters (unavailable)

    Args:
        projections: Combined projections DataFrame
        quality_scores: DataFrame with quality_score per player
        my_roster_names: Set of player names on my roster
        opponent_roster_names: Set of ALL names on ANY opponent roster
        top_n_per_position: Players to keep per position (default 30)
        top_n_per_category: Top players per category to ensure included (default 10)

    Returns:
        Filtered projections DataFrame containing only candidates.
    """
    # Available pool = projections excluding opponent roster
    available_pool = projections[
        ~projections["Name"].isin(opponent_roster_names)
    ].copy()

    # Join quality scores
    available_pool = available_pool.merge(
        quality_scores[["Name", "quality_score"]],
        on="Name",
        how="left",
        suffixes=("", "_qs"),
    )

    # Use existing quality_score if present, otherwise use merged one
    if "quality_score_qs" in available_pool.columns:
        available_pool["quality_score"] = available_pool["quality_score"].fillna(
            available_pool["quality_score_qs"]
        )
        available_pool = available_pool.drop(columns=["quality_score_qs"])

    candidate_names = set()

    # 1. Add all my roster players
    candidate_names.update(my_roster_names)

    # 2. Add top N per slot type
    for slot, valid_positions in SLOT_ELIGIBILITY.items():
        # Find players eligible for this slot
        def is_eligible(pos_str):
            if pd.isna(pos_str):
                return False
            player_positions = set(p.strip() for p in str(pos_str).split(","))
            return bool(player_positions & valid_positions)

        eligible = available_pool[available_pool["Position"].apply(is_eligible)]
        top_players = eligible.nlargest(top_n_per_position, "quality_score")["Name"]
        candidate_names.update(top_players)

    # 3. Add top M per scoring category
    for cat in ALL_CATEGORIES:
        if cat in available_pool.columns:
            if cat in NEGATIVE_CATEGORIES:
                # Lower is better - take players with lowest values
                # Filter to players with some innings first
                if cat in ("ERA", "WHIP"):
                    eligible = available_pool[available_pool["IP"] > 0]
                else:
                    eligible = available_pool
                top_players = eligible.nsmallest(top_n_per_category, cat)["Name"]
            else:
                top_players = available_pool.nlargest(top_n_per_category, cat)["Name"]
            candidate_names.update(top_players)

    # Filter to candidates
    candidates = projections[projections["Name"].isin(candidate_names)].copy()

    # Add quality scores back
    candidates = candidates.merge(
        quality_scores[["Name", "quality_score"]],
        on="Name",
        how="left",
        suffixes=("", "_qs"),
    )
    if "quality_score_qs" in candidates.columns:
        candidates["quality_score"] = candidates["quality_score"].fillna(
            candidates["quality_score_qs"]
        )
        candidates = candidates.drop(columns=["quality_score_qs"])

    hitter_count = (candidates["player_type"] == "hitter").sum()
    pitcher_count = (candidates["player_type"] == "pitcher").sum()

    print(
        f"Filtered to {len(candidates)} candidates from {len(projections)} total players"
    )
    print(f"  - {hitter_count} hitters, {pitcher_count} pitchers")

    return candidates.reset_index(drop=True)


# =============================================================================
# MILP FORMULATION AND SOLVING
# =============================================================================


def build_and_solve_milp(
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    current_roster_names: set[str],
    balance_lambda: float = BALANCE_LAMBDA_DEFAULT,
) -> tuple[list[str], dict]:
    """
    Build and solve the MILP for optimal roster construction.

    Uses a variance-penalized objective that balances total wins against
    category balance:

        maximize  Σ y[j,c] + λ·w_min - λ·w_max

    Where w_min and w_max are the minimum and maximum category win counts.
    This discourages "punting" strategies that abandon some categories.

    Args:
        candidates: DataFrame of candidate players (filtered projections).
                    Must have: Name, Position, player_type, and all stat columns.
        opponent_totals: Dict mapping team_id to category totals.
        current_roster_names: Names currently on my roster (for logging changes).
        balance_lambda: Balance coefficient (default 0.5).
            - 0.0 = standard objective (backward compatible, no balance)
            - 0.5 = moderate balance (recommended)
            - 1.0 = strong balance

    Returns:
        optimal_roster_names: List of player Names for the optimal roster
        solution_info: Dict with:
            - 'objective': float (penalized objective value)
            - 'total_wins': int (raw opponent-category wins, max 60)
            - 'category_wins': dict[str, int] (wins per category, 0-6 each)
            - 'w_min': int (wins in worst category)
            - 'w_max': int (wins in best category)
            - 'win_range': int (w_max - w_min, lower is more balanced)
            - 'solve_time': float (seconds)
            - 'status': str
            - 'balance_lambda': float (λ used)
    """
    assert 0.0 <= balance_lambda <= 2.0, (
        f"balance_lambda must be in [0.0, 2.0], got {balance_lambda}. "
        f"Use 0.0 for standard objective, 0.5 for balanced (recommended)."
    )

    print(f"Building MILP with {len(candidates)} candidates...")

    # Index sets
    I = list(range(len(candidates)))
    I_H = [i for i in I if candidates.iloc[i]["player_type"] == "hitter"]
    I_P = [i for i in I if candidates.iloc[i]["player_type"] == "pitcher"]
    J = list(opponent_totals.keys())  # Opponent team IDs

    # Precompute eligibility
    eligibility = compute_slot_eligibility(candidates)
    validate_slot_coverage(eligibility, candidates)

    # Create problem
    prob = pulp.LpProblem("RosterOptimization", pulp.LpMaximize)

    # === Decision Variables ===

    # Player selection: x[i] = 1 if player i is on roster
    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in I}

    # Slot assignment: a[i,s] = 1 if player i starts in slot type s
    a = {}
    for i in I:
        for s in eligibility[i]:
            a[i, s] = LpVariable(f"a_{i}_{s}", cat="Binary")

    # Beat indicators: y[j,c] = 1 if I beat opponent j in category c
    y = {
        (j, c): LpVariable(f"y_{j}_{c}", cat="Binary")
        for j in J
        for c in ALL_CATEGORIES
    }

    # Balance variables (for variance-penalized objective)
    w_min = LpVariable("w_min", lowBound=0, upBound=NUM_OPPONENTS, cat="Continuous")
    w_max = LpVariable("w_max", lowBound=0, upBound=NUM_OPPONENTS, cat="Continuous")

    var_count = len(x) + len(a) + len(y) + 2  # +2 for w_min, w_max
    print(
        f"  Variables: {len(x)} player, {len(a)} slot, {len(y)} beat, 2 balance ({var_count} total)"
    )

    # === Objective: Variance-penalized total wins ===
    # maximize: Σ y[j,c] + λ·w_min - λ·w_max
    total_wins_expr = lpSum(y[j, c] for j in J for c in ALL_CATEGORIES)
    balance_term = balance_lambda * w_min - balance_lambda * w_max
    prob += total_wins_expr + balance_term

    # === Constraints ===
    constraint_count = 0

    # C1: Roster size
    prob += lpSum(x[i] for i in I) == ROSTER_SIZE, "roster_size"
    constraint_count += 1

    # C2: Slot assignment requires rostering
    for i, s in a:
        prob += a[i, s] <= x[i], f"slot_requires_roster_{i}_{s}"
        constraint_count += 1

    # C3: Each player in at most one slot
    for i in I:
        player_slots = [s for s in eligibility[i] if (i, s) in a]
        if player_slots:
            prob += lpSum(a[i, s] for s in player_slots) <= 1, f"one_slot_{i}"
            constraint_count += 1

    # C4: Starting slots must be filled
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}
    for slot, count in all_slots.items():
        prob += (
            lpSum(a[i, slot] for i in I if slot in eligibility[i]) >= count,
            f"fill_{slot}",
        )
        constraint_count += 1

    # C5: Roster composition bounds
    prob += lpSum(x[i] for i in I_H) >= MIN_HITTERS, "min_hitters"
    prob += lpSum(x[i] for i in I_H) <= MAX_HITTERS, "max_hitters"
    prob += lpSum(x[i] for i in I_P) >= MIN_PITCHERS, "min_pitchers"
    prob += lpSum(x[i] for i in I_P) <= MAX_PITCHERS, "max_pitchers"
    constraint_count += 4

    # C6: Beat constraints for counting stats (R, HR, RBI, SB, W, SV, K)
    counting_hitting = ["R", "HR", "RBI", "SB"]
    counting_pitching = ["W", "SV", "K"]

    for j in J:
        for c in counting_hitting:
            # Use only hitters
            my_sum = lpSum(candidates.iloc[i][c] * x[i] for i in I_H)
            opp_val = opponent_totals[j][c]
            prob += (
                my_sum >= opp_val + EPSILON_COUNTING - BIG_M_COUNTING * (1 - y[j, c]),
                f"beat_{j}_{c}",
            )
            constraint_count += 1

        for c in counting_pitching:
            # Use only pitchers
            my_sum = lpSum(candidates.iloc[i][c] * x[i] for i in I_P)
            opp_val = opponent_totals[j][c]
            prob += (
                my_sum >= opp_val + EPSILON_COUNTING - BIG_M_COUNTING * (1 - y[j, c]),
                f"beat_{j}_{c}",
            )
            constraint_count += 1

    # C7: Beat constraints for OPS (higher is better)
    for j in J:
        opp_ops = opponent_totals[j]["OPS"]
        # Linearized: sum(PA * (OPS - opp_OPS) * x) >= epsilon - M*(1-y)
        coeff_sum = lpSum(
            candidates.iloc[i]["PA"] * (candidates.iloc[i]["OPS"] - opp_ops) * x[i]
            for i in I_H
        )
        prob += (
            coeff_sum >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, "OPS"]),
            f"beat_{j}_OPS",
        )
        constraint_count += 1

    # C8: Beat constraints for ERA and WHIP (lower is better)
    for j in J:
        opp_era = opponent_totals[j]["ERA"]
        # Linearized: sum(IP * (opp_ERA - ERA) * x) >= epsilon - M*(1-y)
        # Note: coefficient is (opponent - player) so positive = good
        coeff_sum = lpSum(
            candidates.iloc[i]["IP"] * (opp_era - candidates.iloc[i]["ERA"]) * x[i]
            for i in I_P
        )
        prob += (
            coeff_sum >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, "ERA"]),
            f"beat_{j}_ERA",
        )
        constraint_count += 1

    for j in J:
        opp_whip = opponent_totals[j]["WHIP"]
        coeff_sum = lpSum(
            candidates.iloc[i]["IP"] * (opp_whip - candidates.iloc[i]["WHIP"]) * x[i]
            for i in I_P
        )
        prob += (
            coeff_sum >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, "WHIP"]),
            f"beat_{j}_WHIP",
        )
        constraint_count += 1

    # C9: w_min bounded above by each category's wins
    for c in ALL_CATEGORIES:
        category_wins_expr = lpSum(y[j, c] for j in J)
        prob += w_min <= category_wins_expr, f"min_bound_{c}"
        constraint_count += 1

    # C10: w_max bounded below by each category's wins
    for c in ALL_CATEGORIES:
        category_wins_expr = lpSum(y[j, c] for j in J)
        prob += w_max >= category_wins_expr, f"max_bound_{c}"
        constraint_count += 1

    print(f"  Constraints: {constraint_count} total")

    # === Solve ===
    print("Solving...")
    start_time = time.time()

    # Try HiGHS first, fall back to CBC
    available_solvers = pulp.listSolvers(onlyAvailable=True)

    if "HiGHS_CMD" in available_solvers:
        solver = pulp.HiGHS_CMD(msg=True, timeLimit=300)
    elif "PULP_CBC_CMD" in available_solvers:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
    else:
        solver = None  # Use default

    status = prob.solve(solver)
    solve_time = time.time() - start_time

    # Check status
    status_str = pulp.LpStatus[status]
    assert status == pulp.LpStatusOptimal, (
        f"Solver failed: {status_str}. "
        f"Check position slot eligibility and roster composition constraints."
    )

    # Extract solution
    roster_names = [candidates.iloc[i]["Name"] for i in I if value(x[i]) > 0.5]

    # Compute balance metrics
    category_wins = {
        c: int(round(sum(value(y[j, c]) for j in J))) for c in ALL_CATEGORIES
    }
    total_wins = sum(category_wins.values())
    actual_w_min = min(category_wins.values())
    actual_w_max = max(category_wins.values())
    win_range = actual_w_max - actual_w_min

    if balance_lambda > 0:
        print(
            f"Solved in {solve_time:.1f}s — {total_wins}/60 wins, "
            f"range {win_range} ({actual_w_min}-{actual_w_max}), λ={balance_lambda}"
        )
    else:
        print(
            f"Solved in {solve_time:.1f}s — objective: {total_wins}/60 opponent-category wins"
        )

    # Log roster changes
    added = set(roster_names) - current_roster_names
    dropped = current_roster_names - set(roster_names)
    if added or dropped:
        print(f"  Added {len(added)} players, dropped {len(dropped)}")

    return roster_names, {
        "objective": pulp.value(prob.objective),
        "total_wins": total_wins,
        "category_wins": category_wins,
        "w_min": actual_w_min,
        "w_max": actual_w_max,
        "win_range": win_range,
        "solve_time": solve_time,
        "status": status_str,
        "balance_lambda": balance_lambda,
    }


# =============================================================================
# STANDINGS AND OUTPUT
# =============================================================================


def compute_standings(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute projected standings for each category.

    Returns:
        DataFrame with columns:
            category, my_value, opp_1, opp_2, ..., opp_6, my_rank, wins

        my_rank: 1 = first place, 7 = last place
        wins: number of opponents I beat (0-6)

    For negative categories (ERA, WHIP), lower value = better rank.
    """
    rows = []

    for cat in ALL_CATEGORIES:
        row = {"category": cat, "my_value": my_totals[cat]}

        # Add opponent values
        all_values = [my_totals[cat]]
        for opp_id in sorted(opponent_totals.keys()):
            row[f"opp_{opp_id}"] = opponent_totals[opp_id][cat]
            all_values.append(opponent_totals[opp_id][cat])

        # Compute rank (1 = best)
        if cat in NEGATIVE_CATEGORIES:
            # Lower is better
            sorted_vals = sorted(all_values)
            row["my_rank"] = sorted_vals.index(my_totals[cat]) + 1
            row["wins"] = sum(1 for v in all_values[1:] if my_totals[cat] < v)
        else:
            # Higher is better
            sorted_vals = sorted(all_values, reverse=True)
            row["my_rank"] = sorted_vals.index(my_totals[cat]) + 1
            row["wins"] = sum(1 for v in all_values[1:] if my_totals[cat] > v)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_roster_change_values(
    added_names: set[str],
    dropped_names: set[str],
    old_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Expected Wins Added (EWA) and SGP for each roster change.

    EWA = change in expected category matchup wins (out of 60 total matchups).
    This is more intuitive and linear than league win probability (V).

    This is used for waiver wire prioritization:
    - Additions sorted by EWA descending (most valuable first)
    - Drops sorted by EWA descending (least harmful to drop first)

    Args:
        added_names: Set of player names being added
        dropped_names: Set of player names being dropped
        old_roster_names: Current roster before changes
        projections: Combined projections DataFrame (must have SGP column)
        opponent_totals: Opponent category totals

    Returns:
        Tuple of (added_df, dropped_df) DataFrames with columns:
            Name, Position, Team, player_type, EWA (expected wins added), SGP

        added_df is sorted by EWA descending (highest value first)
        dropped_df is sorted by EWA descending (negative EWA means loss;
            first entry = closest to 0 = least harmful to drop)
    """
    # Local import to avoid circular dependency
    from .trade_engine import compute_win_probability

    # Compute category sigmas
    old_totals = compute_team_totals(old_roster_names, projections)
    category_sigmas = estimate_projection_uncertainty(old_totals, opponent_totals)

    # Baseline expected wins
    _, baseline_diag = compute_win_probability(
        old_totals, opponent_totals, category_sigmas
    )
    baseline_ew = baseline_diag["expected_wins"]

    # Compute EWA for added players
    # Each add is evaluated in ISOLATION: "what if I just add this one player?"
    # Do NOT remove dropped players - we want to see each add's individual value
    added_rows = []
    for name in added_names:
        player_row = projections[projections["Name"] == name].iloc[0]

        # New roster with just this player added
        test_roster = old_roster_names | {name}

        test_totals = compute_team_totals(test_roster, projections)
        _, test_diag = compute_win_probability(
            test_totals, opponent_totals, category_sigmas
        )
        ewa = test_diag["expected_wins"] - baseline_ew

        added_rows.append(
            {
                "Name": name,
                "Position": player_row["Position"],
                "Team": player_row["Team"],
                "player_type": player_row["player_type"],
                "EWA": ewa,
                "SGP": player_row["SGP"],
            }
        )

    added_df = pd.DataFrame(added_rows).sort_values("EWA", ascending=False)

    # Compute EWA for dropped players
    dropped_rows = []
    for name in dropped_names:
        player_row = projections[projections["Name"] == name].iloc[0]

        # Roster without this player
        test_roster = old_roster_names - {name}
        test_totals = compute_team_totals(test_roster, projections)
        _, test_diag = compute_win_probability(
            test_totals, opponent_totals, category_sigmas
        )
        ewa = (
            test_diag["expected_wins"] - baseline_ew
        )  # Negative means losing them hurts

        dropped_rows.append(
            {
                "Name": name,
                "Position": player_row["Position"],
                "Team": player_row["Team"],
                "player_type": player_row["player_type"],
                "EWA": ewa,
                "SGP": player_row["SGP"],
            }
        )

    dropped_df = pd.DataFrame(dropped_rows).sort_values("EWA", ascending=False)

    return added_df, dropped_df


def print_roster_summary(
    roster_names: list[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    old_roster_names: set[str] | None = None,
    solution_info: dict | None = None,
) -> None:
    """
    Print a formatted summary of the optimal roster.

    Args:
        roster_names: List of player names on the roster
        projections: Combined projections DataFrame
        my_totals: My team's category totals
        opponent_totals: Dict of opponent category totals
        old_roster_names: Previous roster names (for showing changes)
        solution_info: Optional dict from build_and_solve_milp with balance metrics
    """
    roster_df = projections[projections["Name"].isin(roster_names)]
    hitters = roster_df[roster_df["player_type"] == "hitter"]
    pitchers = roster_df[roster_df["player_type"] == "pitcher"]

    print("\n" + "=" * 70)
    print("OPTIMAL ROSTER (26 players)")
    print("=" * 70)

    # Hitters
    print(f"\nHITTERS ({len(hitters)}):")
    print("-" * 60)
    print(
        f"{'Pos':<6} {'Name':<25} {'Team':<5} {'PA':>5} {'R':>4} {'HR':>4} {'RBI':>4} {'SB':>4} {'OPS':>6}"
    )
    print("-" * 60)
    for _, row in hitters.sort_values("SGP", ascending=False).iterrows():
        print(
            f"{row['Position']:<6} {strip_name_suffix(row['Name']):<25} {row['Team']:<5} "
            f"{int(row['PA']):>5} {int(row['R']):>4} {int(row['HR']):>4} "
            f"{int(row['RBI']):>4} {int(row['SB']):>4} {row['OPS']:>6.3f}"
        )

    # Pitchers
    print(f"\nPITCHERS ({len(pitchers)}):")
    print("-" * 60)
    print(
        f"{'Pos':<6} {'Name':<25} {'Team':<5} {'IP':>5} {'W':>4} {'SV':>4} {'K':>4} {'ERA':>6} {'WHIP':>6}"
    )
    print("-" * 60)
    for _, row in pitchers.sort_values("SGP", ascending=False).iterrows():
        print(
            f"{row['Position']:<6} {strip_name_suffix(row['Name']):<25} {row['Team']:<5} "
            f"{row['IP']:>5.0f} {int(row['W']):>4} {int(row['SV']):>4} "
            f"{int(row['K']):>4} {row['ERA']:>6.2f} {row['WHIP']:>6.3f}"
        )

    # Waiver priority list
    if old_roster_names:
        added = set(roster_names) - old_roster_names
        dropped = old_roster_names - set(roster_names)

        if added or dropped:
            print("\n" + "=" * 70)
            print("WAIVER PRIORITY LIST")
            print("=" * 70)

            added_df, dropped_df = compute_roster_change_values(
                added, dropped, old_roster_names, projections, opponent_totals
            )

            if not added_df.empty:
                print(f"\nFREE AGENTS TO ADD (sorted by EWA):")
                print("-" * 60)
                print(
                    f"{'#':<3} {'Name':<25} {'Pos':<6} {'Team':<5} {'EWA':>8} {'SGP':>6}"
                )
                print("-" * 60)
                for i, (_, row) in enumerate(added_df.iterrows(), 1):
                    print(
                        f"{i:<3} {strip_name_suffix(row['Name']):<25} {row['Position']:<6} "
                        f"{row['Team']:<5} {row['EWA']:>+7.2f} {row['SGP']:>6.1f}"
                    )

            if not dropped_df.empty:
                print(f"\nPLAYERS TO DROP (least harmful first):")
                print("-" * 60)
                print(
                    f"{'#':<3} {'Name':<25} {'Pos':<6} {'Team':<5} {'EWA':>8} {'SGP':>6}"
                )
                print("-" * 60)
                for i, (_, row) in enumerate(dropped_df.iterrows(), 1):
                    print(
                        f"{i:<3} {strip_name_suffix(row['Name']):<25} {row['Position']:<6} "
                        f"{row['Team']:<5} {row['EWA']:>+7.2f} {row['SGP']:>6.1f}"
                    )

    # Standings projection
    standings = compute_standings(my_totals, opponent_totals)

    print("\n" + "=" * 70)
    print("STANDINGS PROJECTION")
    print("=" * 70)
    print(f"\n{'Category':<10} {'My Value':>10} {'Rank':>6} {'Wins':>6}")
    print("-" * 35)

    total_wins = 0
    total_roto_points = 0
    for _, row in standings.iterrows():
        cat = row["category"]
        val = row["my_value"]
        if cat in ["ERA", "WHIP", "OPS"]:
            val_str = f"{val:.3f}"
        else:
            val_str = f"{int(val)}"

        print(f"{cat:<10} {val_str:>10} {int(row['my_rank']):>6} {int(row['wins']):>6}")
        total_wins += row["wins"]
        total_roto_points += 8 - row["my_rank"]  # 7 for 1st, 6 for 2nd, etc.

    print("-" * 35)
    print(f"\nTotal opponent-category wins: {int(total_wins)} / 60")
    print(f"Projected roto points: {int(total_roto_points)} / 70")

    # Balance metrics (if available)
    if solution_info and solution_info.get("balance_lambda", 0) > 0:
        print("\n" + "-" * 35)
        print(f"BALANCE METRICS (λ = {solution_info['balance_lambda']})")
        print("-" * 35)

        cat_wins = solution_info.get("category_wins", {})
        if cat_wins:
            w_min = solution_info["w_min"]
            w_max = solution_info["w_max"]

            worst_cats = [c for c, w in cat_wins.items() if w == w_min]
            best_cats = [c for c, w in cat_wins.items() if w == w_max]

            print(f"Worst: {', '.join(worst_cats)} ({w_min} wins)")
            print(f"Best:  {', '.join(best_cats)} ({w_max} wins)")
            print(f"Range: {solution_info['win_range']}")


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================


def compute_position_sensitivity(
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict:
    """
    Compute position-by-position sensitivity analysis.

    This answers: "What's the Expected Wins Added (EWA) from upgrading at each position?"

    For each position slot:
    1. Identify all eligible players and mark status (my_roster, opponent, available)
    2. Compute baseline expected wins
    3. For each available candidate, compute EWA if they replaced my worst player

    Args:
        my_roster_names: Set of player names on my roster
        opponent_roster_names: Set of ALL player names on ANY opponent roster
        projections: Full projections DataFrame with SGP
        opponent_totals: Dict mapping team_id to category totals
        category_sigmas: Standard deviations per category

    Returns:
        Dict with keys:
            - 'slot_data': Dict mapping slot name to DataFrame of eligible players
                           with columns: Name, SGP, percentile, status
            - 'ewa_df': DataFrame with swap scenarios and EWA values
            - 'sensitivity_df': DataFrame with EWA per SGP for each position
            - 'baseline_expected_wins': float
            - 'baseline_roto_points': int
    """
    from .trade_engine import compute_win_probability

    # Compute baseline
    my_totals = compute_team_totals(my_roster_names, projections)
    _, baseline_diagnostics = compute_win_probability(
        my_totals, opponent_totals, category_sigmas
    )
    baseline_ew = baseline_diagnostics["expected_wins"]
    baseline_roto = baseline_diagnostics["expected_roto_points"]

    print(f"Baseline expected wins: {baseline_ew:.1f} / 60")

    # Build slot data with proper status marking
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}
    slot_data = {}

    # Minimum playing time thresholds for FA filtering
    # (keeps all rostered players regardless of PA/IP)
    MIN_PA_FOR_FA = 50  # ~1/3 of a full season
    MIN_IP_FOR_FA = 20  # ~1/3 of a starter's workload

    for slot, valid_positions in SLOT_ELIGIBILITY.items():
        if slot not in all_slots:
            continue

        # Filter to eligible players using vectorized string operations
        def check_eligibility(pos_str):
            if pd.isna(pos_str):
                return False
            positions = set(p.strip() for p in str(pos_str).split(","))
            return bool(positions & valid_positions)

        eligible = projections[projections["Position"].apply(check_eligibility)].copy()

        # Mark status: my_roster > opponent > available (priority order)
        eligible["status"] = "available"
        eligible.loc[eligible["Name"].isin(opponent_roster_names), "status"] = (
            "opponent"
        )
        eligible.loc[eligible["Name"].isin(my_roster_names), "status"] = "my_roster"

        # Filter out low-PA/IP free agents (keep all rostered players)
        # This dramatically reduces computation for UTIL (4000+ → ~500) and RP (4000+ → ~400)
        is_rostered = eligible["status"].isin(["my_roster", "opponent"])
        if slot in HITTING_SLOTS:
            has_meaningful_pt = eligible["PA"] >= MIN_PA_FOR_FA
        else:
            has_meaningful_pt = eligible["IP"] >= MIN_IP_FOR_FA
        eligible = eligible[is_rostered | has_meaningful_pt].copy()

        # Sort by SGP descending and compute rank/percentile
        eligible = eligible.sort_values("SGP", ascending=False).reset_index(drop=True)
        eligible["rank"] = np.arange(1, len(eligible) + 1)
        eligible["percentile"] = 100.0 * (1 - eligible["rank"] / len(eligible))

        slot_data[slot] = eligible

    # Compute EWA for upgrade scenarios
    print("Computing position sensitivities...")
    ewa_results = []

    for slot in tqdm(all_slots.keys(), desc="Positions"):
        eligible = slot_data[slot]
        my_at_slot = eligible[eligible["status"] == "my_roster"]
        available = eligible[eligible["status"] == "available"]

        if my_at_slot.empty or len(available) < 5:
            continue

        # Get my worst player at this slot (lowest SGP among my players)
        my_worst_idx = my_at_slot["SGP"].idxmin()
        my_worst = my_at_slot.loc[my_worst_idx]

        # Sample available players at different quality levels (rank 1, 5, 10, 20, 50)
        available_sorted = available.nlargest(50, "SGP")
        sample_ranks = [1, 5, 10, 20, 50]

        for rank in sample_ranks:
            if rank > len(available_sorted):
                continue
            candidate = available_sorted.iloc[rank - 1]

            # Compute EWA for this swap
            new_roster = (my_roster_names - {my_worst["Name"]}) | {candidate["Name"]}
            new_totals = compute_team_totals(new_roster, projections)
            _, new_diagnostics = compute_win_probability(
                new_totals, opponent_totals, category_sigmas
            )

            ewa_results.append(
                {
                    "slot": slot,
                    "my_player": my_worst["Name"],
                    "my_sgp": my_worst["SGP"],
                    "my_pctl": my_worst["percentile"],
                    "candidate": candidate["Name"],
                    "candidate_sgp": candidate["SGP"],
                    "candidate_pctl": candidate["percentile"],
                    "candidate_rank": rank,
                    "sgp_delta": candidate["SGP"] - my_worst["SGP"],
                    "new_expected_wins": new_diagnostics["expected_wins"],
                    "ewa": new_diagnostics["expected_wins"] - baseline_ew,
                }
            )

    ewa_df = pd.DataFrame(ewa_results)

    # Compute sensitivity (EWA per SGP) for each slot
    sensitivity_rows = []
    for slot in all_slots.keys():
        slot_ewa = ewa_df[ewa_df["slot"] == slot]
        if len(slot_ewa) < 2:
            continue

        # Only use positive SGP deltas (actual upgrades)
        upgrades = slot_ewa[slot_ewa["sgp_delta"] > 0]
        if len(upgrades) < 2:
            continue

        # EWA per SGP = mean(EWA) / mean(SGP delta)
        avg_ewa_per_sgp = upgrades["ewa"].mean() / upgrades["sgp_delta"].mean()

        # Get my players and available players at this slot
        eligible = slot_data[slot]
        my_at_slot = eligible[eligible["status"] == "my_roster"]
        available = eligible[eligible["status"] == "available"]

        if my_at_slot.empty:
            continue

        # My worst player's SGP
        my_worst_sgp = my_at_slot["SGP"].min()
        my_worst_name = my_at_slot.loc[my_at_slot["SGP"].idxmin(), "Name"]

        # Count how many available players are BETTER than my worst
        better_fas = available[available["SGP"] > my_worst_sgp]
        better_fas_count = len(better_fas)

        # Best available FA SGP gap
        best_fa_sgp = available["SGP"].max() if not available.empty else my_worst_sgp
        best_fa_sgp_gap = best_fa_sgp - my_worst_sgp

        # Best FA EWA (from ewa_df)
        best_fa_ewa = upgrades["ewa"].max() if not upgrades.empty else 0.0

        sensitivity_rows.append(
            {
                "slot": slot,
                "ewa_per_sgp": avg_ewa_per_sgp,
                "my_worst_name": my_worst_name,
                "my_worst_sgp": my_worst_sgp,
                "better_fas_count": better_fas_count,
                "best_fa_sgp_gap": best_fa_sgp_gap,
                "best_fa_ewa": best_fa_ewa,
            }
        )

    sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values(
        "best_fa_ewa", ascending=False
    )

    return {
        "slot_data": slot_data,
        "ewa_df": ewa_df,
        "sensitivity_df": sensitivity_df,
        "baseline_expected_wins": baseline_ew,
        "baseline_roto_points": baseline_roto,
    }


def compute_percentile_sensitivity(
    my_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    slot_data: dict[str, pd.DataFrame],
    baseline_expected_wins: float,
    include_all_active: bool = True,
) -> pd.DataFrame:
    """
    Compute EWA at different percentile levels for each position.

    Players are ranked by EWA (not SGP), so x=EWA percentile, y=EWA is monotonic.
    This answers: "What's the slope of the EWA curve at position X?"

    Args:
        my_roster_names: Set of player names on my roster
        projections: Full projections DataFrame
        opponent_totals: Dict mapping team_id to category totals
        category_sigmas: Standard deviations per category
        slot_data: Output from compute_position_sensitivity
        baseline_expected_wins: Baseline expected wins (avoids recomputation)
        include_all_active: If True, include opponent roster players (for trade context).
                           If False, only include available (FA) players.
                           Default: True

    Returns:
        DataFrame with columns: slot, ewa_pctl, candidate, candidate_sgp, ewa, status, my_worst_ewa_pctl
        (percentile is computed among active players ranked by EWA)
    """
    from .trade_engine import compute_win_probability

    percentile_ewa = []
    target_percentiles = [95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]

    for slot, eligible in tqdm(slot_data.items(), desc="EWA percentile analysis"):
        my_at_slot = eligible[eligible["status"] == "my_roster"]

        if my_at_slot.empty:
            continue

        # Get active players (exclude my roster for swap analysis)
        if include_all_active:
            # Include opponent roster players (for trade context)
            active = eligible[eligible["status"] != "my_roster"].copy()
        else:
            # Only available (FA) players
            active = eligible[eligible["status"] == "available"].copy()

        if len(active) < 20:
            continue

        # Get my worst player at this slot (lowest SGP)
        my_worst_idx = my_at_slot["SGP"].idxmin()
        my_worst = my_at_slot.loc[my_worst_idx]

        # Compute EWA for every active player (swap my worst → candidate)
        print(f"  {slot}: computing EWA for {len(active)} players...")
        ewa_list = []
        for _, candidate in active.iterrows():
            new_roster = (my_roster_names - {my_worst["Name"]}) | {candidate["Name"]}
            new_totals = compute_team_totals(new_roster, projections)
            _, new_diagnostics = compute_win_probability(
                new_totals, opponent_totals, category_sigmas
            )
            ewa_list.append(new_diagnostics["expected_wins"] - baseline_expected_wins)

        active["ewa"] = ewa_list

        # Rank by EWA descending (highest EWA = rank 1)
        active = active.sort_values("ewa", ascending=False).reset_index(drop=True)
        n_active = len(active)
        active["ewa_pctl"] = 100.0 * (1 - np.arange(n_active) / n_active)

        # Compute my worst player's EWA percentile (where would they rank if traded in?)
        my_worst_ewa = 0.0  # By definition, swapping my worst for my worst = 0 EWA
        better_ewa_count = (active["ewa"] > my_worst_ewa).sum()
        my_worst_ewa_pctl = 100.0 * (1 - better_ewa_count / n_active)

        # Sample at target percentiles
        for target_pctl in target_percentiles:
            pctl_diffs = np.abs(active["ewa_pctl"].values - target_pctl)
            closest_idx = pctl_diffs.argmin()
            closest = active.iloc[closest_idx]

            percentile_ewa.append(
                {
                    "slot": slot,
                    "ewa_pctl": closest["ewa_pctl"],
                    "target_pctl": target_pctl,
                    "candidate": closest["Name"],
                    "candidate_sgp": closest["SGP"],
                    "ewa": closest["ewa"],
                    "status": closest["status"],
                    "my_worst_ewa_pctl": my_worst_ewa_pctl,
                }
            )

    return pd.DataFrame(percentile_ewa)
