"""
Trade Engine using probabilistic win model.

Based on Rosenof (2025) "Optimizing for Rotisserie Fantasy Basketball" arXiv:2501.00933.

Answers: "Which trades improve my chances of winning the rotisserie league
while being fair enough for opponents to accept?"
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from .config import (
    SGP_METRIC,
    TRADE_FAIRNESS_THRESHOLD_PERCENT,
    TRADE_LOSE_COST_SCALE,
    TRADE_MAX_SIZE,
    TRADE_MIN_MEANINGFUL_IMPROVEMENT,
    MIN_STAT_STANDARD_DEVIATION,
)
from .data_loader import (
    ALL_CATEGORIES,
    HITTING_CATEGORIES,
    MAX_HITTERS,
    MAX_PITCHERS,
    MIN_HITTERS,
    MIN_PITCHERS,
    NEGATIVE_CATEGORIES,
    NUM_OPPONENTS,
    PITCHING_CATEGORIES,
    RATIO_STATS,
    compute_team_totals,
    estimate_projection_uncertainty,
    strip_name_suffix,
)

# =============================================================================
# TRADE ENGINE CONFIGURATION
# =============================================================================

# Expected value and variance of maximum of N standard normals
# From Teichroew (1956), used by Rosenof (2025)
MEV_TABLE = {1: 0.0, 2: 0.564, 3: 0.846, 4: 1.029, 5: 1.163, 6: 1.267}
MVAR_TABLE = {1: 1.0, 2: 0.682, 3: 0.559, 4: 0.492, 5: 0.448, 6: 0.416}

# Trade engine configuration loaded from config.json
FAIRNESS_THRESHOLD_PERCENT = TRADE_FAIRNESS_THRESHOLD_PERCENT
MAX_TRADE_SIZE = TRADE_MAX_SIZE
MIN_MEANINGFUL_IMPROVEMENT = TRADE_MIN_MEANINGFUL_IMPROVEMENT
MIN_STD = MIN_STAT_STANDARD_DEVIATION
LOSE_COST_SCALE = TRADE_LOSE_COST_SCALE


# =============================================================================
# GAP AND PROBABILITY COMPUTATION
# =============================================================================


def compute_normalized_gaps(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict[str, dict[int, float]]:
    """
    Compute normalized gaps for each (category, opponent) pair.

    The normalized gap μ_c,o is:
        (my_total[c] - opp_total[o][c]) / (σ_c * √2)

    For NEGATIVE_CATEGORIES (ERA, WHIP), flip the sign so positive = winning.

    The √2 factor ensures the difference of two team totals has unit variance
    when each team's total has variance σ_c².

    Returns:
        Dict[category, Dict[opponent_id, normalized_gap]]
    """
    gaps = {}

    for cat in ALL_CATEGORIES:
        gaps[cat] = {}
        sigma = max(category_sigmas[cat], MIN_STD)

        for opp_id, opp_totals in opponent_totals.items():
            if cat in NEGATIVE_CATEGORIES:
                # Lower is better - flip sign
                raw_gap = opp_totals[cat] - my_totals[cat]
            else:
                raw_gap = my_totals[cat] - opp_totals[cat]

            gaps[cat][opp_id] = raw_gap / (sigma * np.sqrt(2))

    return gaps


def compute_matchup_probabilities(
    normalized_gaps: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    """
    Compute probability of winning each matchup.

    P(beat opponent o in category c) = Φ(μ_c,o)

    Returns:
        Dict[category, Dict[opponent_id, probability]]
        All values in [0, 1].
    """
    probs = {}

    for cat in ALL_CATEGORIES:
        probs[cat] = {}
        for opp_id, gap in normalized_gaps[cat].items():
            probs[cat][opp_id] = stats.norm.cdf(gap)

    return probs


# =============================================================================
# WIN PROBABILITY COMPUTATION
# =============================================================================


def compute_win_probability(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    category_correlations: dict[tuple[str, str], float] | None = None,
) -> tuple[float, dict]:
    """
    Compute probability of winning the rotisserie league.

    Implements the Rosenof (2025) tractable approximation.

    Args:
        my_totals: My team's category totals
        opponent_totals: Opponent totals (6 opponents)
        category_sigmas: Standard deviation per category
        category_correlations: Optional correlations between categories
                               (default: assume independence)

    Returns:
        V: Victory probability (0 to 1)
        diagnostics: Dict with intermediate values
    """
    n_opponents = len(opponent_totals)
    n_categories = len(ALL_CATEGORIES)

    # Step 1: Compute normalized gaps
    normalized_gaps = compute_normalized_gaps(
        my_totals, opponent_totals, category_sigmas
    )

    # Step 2: Compute matchup probabilities
    matchup_probs = compute_matchup_probabilities(normalized_gaps)

    # Step 3: Compute μ_T (expected fantasy points)
    mu_T = 0.0
    for cat in ALL_CATEGORIES:
        for opp_id in opponent_totals.keys():
            mu_T += matchup_probs[cat][opp_id]

    # Step 4: Compute σ_T² (variance of fantasy points)
    # Base variance from matchup outcomes
    sigma_T_sq = 0.0
    for cat in ALL_CATEGORIES:
        for opp_id in opponent_totals.keys():
            p = matchup_probs[cat][opp_id]
            sigma_T_sq += p * (1 - p)  # Bernoulli variance

    # Add correlation adjustment if provided
    # For now, assume independence (no adjustment)

    # Step 5: Compute expected variance of opponent scores (E[σ_M²])
    # Assume similar variance structure for opponents
    E_sigma_M_sq = sigma_T_sq  # Simplified assumption

    # Step 6: Compute μ_L and σ_L (target to beat)
    mev = MEV_TABLE.get(n_opponents, MEV_TABLE[6])
    mvar = MVAR_TABLE.get(n_opponents, MVAR_TABLE[6])

    mu_L = mev * np.sqrt(max(E_sigma_M_sq, MIN_STD))
    sigma_L_sq = E_sigma_M_sq * mvar

    # Step 7: Compute differential distribution
    # μ_D = μ_T * (|O|+1)/|O| - |C|*(|O|+1)/2 - μ_L
    scaling = (n_opponents + 1) / n_opponents
    mu_D = mu_T * scaling - n_categories * (n_opponents + 1) / 2 - mu_L

    # σ_D² = ((|O|+1)/|O|) * σ_T² + σ_L²
    sigma_D_sq = scaling * sigma_T_sq + sigma_L_sq
    sigma_D = np.sqrt(max(sigma_D_sq, MIN_STD))

    # Step 8: Compute V = Φ(μ_D / σ_D)
    V = stats.norm.cdf(mu_D / sigma_D)

    # Compute expected roto points (rough estimate)
    expected_roto_points = 0
    for cat in ALL_CATEGORIES:
        avg_prob = np.mean([matchup_probs[cat][o] for o in opponent_totals.keys()])
        # Estimate rank: 7 * (1 - avg_prob) + 1 → roto points = 7 - rank + 1
        expected_rank = int(7 * (1 - avg_prob) + 1)
        expected_roto_points += max(0, 8 - expected_rank)

    diagnostics = {
        "normalized_gaps": normalized_gaps,
        "matchup_probs": matchup_probs,
        "mu_T": mu_T,
        "sigma_T_sq": sigma_T_sq,
        "mu_L": mu_L,
        "sigma_L_sq": sigma_L_sq,
        "mu_D": mu_D,
        "sigma_D": sigma_D,
        "expected_wins": mu_T,
        "expected_roto_points": expected_roto_points,
    }

    return V, diagnostics


# =============================================================================
# PLAYER VALUE COMPUTATION
# =============================================================================


def _compute_totals_with_player_change(
    my_totals: dict[str, float],
    my_roster_names: set[str],
    add_player: str | None,
    remove_player: str | None,
    projections: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute new team totals after adding/removing a player.

    Simply modifies the roster set and calls compute_team_totals.
    """
    new_roster = my_roster_names.copy()

    if remove_player:
        new_roster.discard(remove_player)
    if add_player:
        new_roster.add(add_player)

    return compute_team_totals(new_roster, projections)


def compute_player_marginal_value(
    player_name: str,
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    my_roster_names: set[str],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    is_acquisition: bool = True,
) -> tuple[float, dict]:
    """
    Compute the marginal change in expected wins from acquiring/losing a player.

    Uses numerical differentiation (actual expected_wins computation before/after).

    Returns:
        ewa: Expected Wins Added (change in expected category matchup wins out of 60)
        breakdown: Dict with V_before, V_after, ew_before, ew_after
    """
    _, diag_before = compute_win_probability(
        my_totals, opponent_totals, category_sigmas
    )
    ew_before = diag_before["expected_wins"]

    if is_acquisition:
        # Adding this player to roster
        new_totals = _compute_totals_with_player_change(
            my_totals,
            my_roster_names,
            add_player=player_name,
            remove_player=None,
            projections=projections,
        )
    else:
        # Losing this player from roster
        new_totals = _compute_totals_with_player_change(
            my_totals,
            my_roster_names,
            add_player=None,
            remove_player=player_name,
            projections=projections,
        )

    _, diag_after = compute_win_probability(
        new_totals, opponent_totals, category_sigmas
    )
    ew_after = diag_after["expected_wins"]

    ewa = ew_after - ew_before

    # Include both V and expected_wins in breakdown for flexibility
    breakdown = {
        "ew_before": ew_before,
        "ew_after": ew_after,
    }

    return ewa, breakdown


def compute_player_values(
    player_names: set[str],
    my_roster_names: set[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> pd.DataFrame:
    """
    Compute marginal value (EWA) for a set of players.

    EWA = Expected Wins Added (change in expected category matchup wins out of 60).

    Returns:
        DataFrame with columns:
            Name, player_type, Position, on_my_roster,
            ewa_acquire (EWA if I acquire this player),
            ewa_lose (EWA if I lose this player, NaN if not on my roster),
            generic_value (SGP for trade fairness)
    """
    results = []

    for player_name in tqdm(player_names, desc="Computing player values"):
        player_row = projections[projections["Name"] == player_name]
        if player_row.empty:
            continue
        player_row = player_row.iloc[0]

        on_roster = player_name in my_roster_names

        # Compute EWA for acquisition
        ewa_acquire, _ = compute_player_marginal_value(
            player_name,
            projections,
            my_totals,
            my_roster_names,
            opponent_totals,
            category_sigmas,
            is_acquisition=True,
        )

        # Compute EWA for losing (only if on roster)
        if on_roster:
            ewa_lose, _ = compute_player_marginal_value(
                player_name,
                projections,
                my_totals,
                my_roster_names,
                opponent_totals,
                category_sigmas,
                is_acquisition=False,
            )
        else:
            ewa_lose = np.nan

        # Generic value: use SGP metric from config (raw or dynasty)
        sgp_column = "dynasty_SGP" if SGP_METRIC == "dynasty" else "SGP"
        # Fall back to SGP if dynasty_SGP not available
        if sgp_column not in player_row or pd.isna(player_row[sgp_column]):
            generic_value = player_row["SGP"]
        else:
            generic_value = player_row[sgp_column]

        results.append(
            {
                "Name": player_name,
                "player_type": player_row["player_type"],
                "Position": player_row["Position"],
                "on_my_roster": on_roster,
                "ewa_acquire": ewa_acquire,
                "ewa_lose": ewa_lose,
                "generic_value": generic_value,
            }
        )

    df = pd.DataFrame(results)
    return df.sort_values("ewa_acquire", ascending=False)


# =============================================================================
# TRADE CANDIDATE IDENTIFICATION
# =============================================================================


def identify_trade_targets(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    opponent_rosters: dict[int, set[str]],
    n_targets: int = 15,
) -> pd.DataFrame:
    """
    Identify players to TARGET (acquire) in trades.

    Targets are players on opponent rosters ranked by EWA (expected wins added).
    """
    # Find which opponent owns each player
    owner_map = {}
    for opp_id, roster in opponent_rosters.items():
        for name in roster:
            owner_map[name] = opp_id

    # Filter to opponent players with positive value
    targets = player_values[~player_values["on_my_roster"]].copy()
    targets = targets[targets["Name"].isin(owner_map.keys())]
    targets = targets[targets["ewa_acquire"] > 0.01]  # Only positive value players

    # Sort by ewa_acquire (how much they help us)
    targets = targets.sort_values("ewa_acquire", ascending=False).head(n_targets)

    # Ensure at least 40% hitters and 40% pitchers
    hitters = targets[targets["player_type"] == "hitter"]
    pitchers = targets[targets["player_type"] == "pitcher"]

    min_each = int(n_targets * 0.4)
    if len(hitters) < min_each:
        # Add more hitters
        more_hitters = player_values[
            (~player_values["on_my_roster"])
            & (player_values["Name"].isin(owner_map.keys()))
            & (player_values["player_type"] == "hitter")
            & (~player_values["Name"].isin(targets["Name"]))
        ].head(min_each - len(hitters))
        targets = pd.concat([targets, more_hitters])

    if len(pitchers) < min_each:
        more_pitchers = player_values[
            (~player_values["on_my_roster"])
            & (player_values["Name"].isin(owner_map.keys()))
            & (player_values["player_type"] == "pitcher")
            & (~player_values["Name"].isin(targets["Name"]))
        ].head(min_each - len(pitchers))
        targets = pd.concat([targets, more_pitchers])

    # Add owner info AFTER all concatenations (so all players get mapped)
    targets["owner_id"] = targets["Name"].map(owner_map)

    if len(targets) > 0:
        best = targets.iloc[0]
        print(f"Trade targets: {len(targets)} players identified (ranked by EWA)")
        print(
            f"  Best: {strip_name_suffix(best['Name'])} from Team {best.get('owner_id', '?')} "
            f"(EWA: +{best['ewa_acquire']:.2f}, SGP: {best['generic_value']:.1f})"
        )

    return targets


def identify_trade_pieces(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    n_pieces: int = 15,
) -> pd.DataFrame:
    """
    Identify players to OFFER (trade away) from my roster.

    Good trade pieces have:
        - High SGP (attractive to opponents)
        - Low ewa_lose (not critical to MY expected wins)
    """
    # Filter to my roster
    pieces = player_values[player_values["on_my_roster"]].copy()

    # Compute expendability (higher = more expendable, easier to trade away)
    # - Low SGP: easier to get fair value from opponent
    # - Small lose cost (ewa_lose close to 0): doesn't hurt us to lose them
    # Since ewa_lose is negative when losing hurts, we ADD it (making expendability lower)
    pieces["expendability"] = (
        -pieces["generic_value"] + pieces["ewa_lose"].fillna(0) * LOSE_COST_SCALE
    )

    # Sort by expendability (most expendable first)
    pieces = pieces.sort_values("expendability", ascending=False).head(n_pieces)

    # Ensure mix of player types
    hitters = pieces[pieces["player_type"] == "hitter"]
    pitchers = pieces[pieces["player_type"] == "pitcher"]

    min_each = int(n_pieces * 0.4)
    if len(hitters) < min_each:
        more_hitters = player_values[
            (player_values["on_my_roster"])
            & (player_values["player_type"] == "hitter")
            & (~player_values["Name"].isin(pieces["Name"]))
        ].head(min_each - len(hitters))
        pieces = pd.concat([pieces, more_hitters])

    if len(pitchers) < min_each:
        more_pitchers = player_values[
            (player_values["on_my_roster"])
            & (player_values["player_type"] == "pitcher")
            & (~player_values["Name"].isin(pieces["Name"]))
        ].head(min_each - len(pitchers))
        pieces = pd.concat([pieces, more_pitchers])

    if len(pieces) > 0:
        best = pieces.iloc[0]
        print(f"Trade pieces: {len(pieces)} players identified")
        print(
            f"  Most expendable: {strip_name_suffix(best['Name'])} "
            f"(SGP={best['generic_value']:.1f}, lose_cost={best['ewa_lose']:.2f})"
        )

    return pieces


# =============================================================================
# TRADE EVALUATION
# =============================================================================


def evaluate_trade(
    send_players: list[str],
    receive_players: list[str],
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    opponent_rosters: dict[int, set[str]] | None = None,
    fairness_threshold: float = FAIRNESS_THRESHOLD_PERCENT,
    min_improvement: float = MIN_MEANINGFUL_IMPROVEMENT,
) -> dict:
    """
    Evaluate a specific trade proposal.

    Args:
        fairness_threshold: Max SGP differential as fraction of total SGP (default 0.10 = 10%)
        min_improvement: Min win probability change to recommend ACCEPT (default 0.001 = 0.1%)
    """
    # Validation
    send_set = set(send_players)
    receive_set = set(receive_players)

    assert send_set <= my_roster_names, (
        f"Cannot send players not on roster: {send_set - my_roster_names}"
    )
    assert not (send_set & receive_set), (
        f"Cannot send and receive same player: {send_set & receive_set}"
    )

    proj_names = set(projections["Name"])
    missing = receive_set - proj_names
    assert not missing, f"Received players not in projections: {missing}"

    # Compute new roster
    new_roster = (my_roster_names - send_set) | receive_set

    # Validate composition - return invalid trade if bounds violated
    new_roster_df = projections[projections["Name"].isin(new_roster)]
    n_hitters = (new_roster_df["player_type"] == "hitter").sum()
    n_pitchers = (new_roster_df["player_type"] == "pitcher").sum()

    if not (MIN_HITTERS <= n_hitters <= MAX_HITTERS):
        return {
            "send_players": send_players,
            "receive_players": receive_players,
            "ewa": float("-inf"),
            "delta_generic": 0.0,
            "ew_before": 0.0,
            "ew_after": 0.0,
            "is_fair": False,
            "is_good_for_me": False,
            "recommendation": "INVALID",
            "category_impact": {},
            "invalid_reason": f"Post-trade hitters ({n_hitters}) outside bounds [{MIN_HITTERS}, {MAX_HITTERS}]",
        }
    if not (MIN_PITCHERS <= n_pitchers <= MAX_PITCHERS):
        return {
            "send_players": send_players,
            "receive_players": receive_players,
            "ewa": float("-inf"),
            "delta_generic": 0.0,
            "ew_before": 0.0,
            "ew_after": 0.0,
            "is_fair": False,
            "is_good_for_me": False,
            "recommendation": "INVALID",
            "category_impact": {},
            "invalid_reason": f"Post-trade pitchers ({n_pitchers}) outside bounds [{MIN_PITCHERS}, {MAX_PITCHERS}]",
        }

    # Compute expected wins before and after
    _, diag_before = compute_win_probability(
        my_totals, opponent_totals, category_sigmas
    )
    ew_before = diag_before["expected_wins"]

    new_totals = compute_team_totals(new_roster, projections)
    _, diag_after = compute_win_probability(
        new_totals, opponent_totals, category_sigmas
    )
    ew_after = diag_after["expected_wins"]

    ewa = ew_after - ew_before

    # Compute dynasty_SGP change
    send_sgp = sum(
        player_values[player_values["Name"] == p]["generic_value"].iloc[0]
        for p in send_players
        if p in player_values["Name"].values
    )
    receive_sgp = sum(
        player_values[player_values["Name"] == p]["generic_value"].iloc[0]
        for p in receive_players
        if p in player_values["Name"].values
    )

    # Handle case where players aren't in player_values
    if send_sgp == 0:
        send_sgp = sum(
            projections[projections["Name"] == p]["SGP"].iloc[0] for p in send_players
        )
    if receive_sgp == 0:
        receive_sgp = sum(
            projections[projections["Name"] == p]["SGP"].iloc[0]
            for p in receive_players
        )

    delta_generic = receive_sgp - send_sgp

    # Fairness check (percentage-based)
    total_sgp = send_sgp + receive_sgp
    if total_sgp > 0:
        relative_diff = abs(delta_generic) / total_sgp
        is_fair = relative_diff <= fairness_threshold
    else:
        is_fair = True

    # Determine recommendation (min_improvement is now in EWA terms, e.g., 0.1 expected wins)
    is_good_for_me = ewa >= min_improvement
    is_bad_for_me = ewa <= -min_improvement

    if is_good_for_me and is_fair:
        recommendation = "ACCEPT"
    elif is_good_for_me and not is_fair and delta_generic > 0:
        recommendation = "STEAL"  # I'm getting more SGP
    elif is_bad_for_me and is_fair:
        recommendation = "REJECT"
    elif not is_fair and is_bad_for_me:
        recommendation = "UNFAIR"
    else:
        recommendation = "NEUTRAL"

    # Find trade partner
    trade_partner_id = None
    if opponent_rosters:
        for opp_id, roster in opponent_rosters.items():
            if receive_set <= roster:
                trade_partner_id = opp_id
                break

    # Category impact
    category_impact = {cat: new_totals[cat] - my_totals[cat] for cat in ALL_CATEGORIES}

    # Build result
    result = {
        "send_players": send_players,
        "receive_players": receive_players,
        "ewa": ewa,
        "delta_generic": delta_generic,
        "ew_before": ew_before,
        "ew_after": ew_after,
        "is_fair": is_fair,
        "is_good_for_me": is_good_for_me,
        "recommendation": recommendation,
        "category_impact": category_impact,
        "send_generics": [(p, send_sgp / len(send_players)) for p in send_players],
        "receive_generics": [
            (p, receive_sgp / len(receive_players)) for p in receive_players
        ],
        "trade_partner_id": trade_partner_id,
    }

    return result


def generate_trade_candidates(
    my_roster_names: set[str],
    player_values: pd.DataFrame,
    opponent_rosters: dict[int, set[str]],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    max_send: int = 2,
    max_receive: int = 2,
    n_targets: int = 15,
    n_pieces: int = 15,
    n_candidates: int = 20,
    fairness_threshold: float = FAIRNESS_THRESHOLD_PERCENT,
    min_improvement: float = MIN_MEANINGFUL_IMPROVEMENT,
) -> list[dict]:
    """
    Generate candidate trades to consider.

    Args:
        fairness_threshold: Max SGP differential as fraction of total SGP (default 0.10 = 10%)
        min_improvement: Min win probability change to recommend ACCEPT (default 0.001 = 0.1%)
    """
    # Identify targets and pieces
    targets = identify_trade_targets(
        player_values, my_roster_names, opponent_rosters, n_targets
    )
    pieces = identify_trade_pieces(player_values, my_roster_names, n_pieces)

    if targets.empty or pieces.empty:
        print("No trade targets or pieces available")
        return []

    # Group targets by owner
    owner_targets = {}
    for _, row in targets.iterrows():
        owner_id = row.get("owner_id")
        if owner_id is not None:
            if owner_id not in owner_targets:
                owner_targets[owner_id] = []
            owner_targets[owner_id].append(row["Name"])

    piece_names = list(pieces["Name"])

    # Generate combinations
    all_trades = []
    combos_evaluated = 0

    for send_size in range(1, min(max_send, len(piece_names)) + 1):
        for receive_size in range(1, min(max_receive + 1, 4)):
            # For each owner
            for owner_id, owner_target_list in owner_targets.items():
                if len(owner_target_list) < receive_size:
                    continue

                # Generate send combinations
                for send_combo in combinations(piece_names, send_size):
                    # Generate receive combinations from this owner
                    for receive_combo in combinations(owner_target_list, receive_size):
                        combos_evaluated += 1

                        # Evaluate trade
                        trade_result = evaluate_trade(
                            list(send_combo),
                            list(receive_combo),
                            player_values,
                            my_roster_names,
                            projections,
                            my_totals,
                            opponent_totals,
                            category_sigmas,
                            opponent_rosters,
                            fairness_threshold=fairness_threshold,
                            min_improvement=min_improvement,
                        )

                        # Keep only fair + good trades
                        if trade_result["is_fair"] and trade_result["is_good_for_me"]:
                            all_trades.append(trade_result)

    # Sort by EWA descending
    all_trades.sort(key=lambda t: t["ewa"], reverse=True)

    # Keep top N
    all_trades = all_trades[:n_candidates]

    print(f"Generated {len(all_trades)} candidate trades")
    print(f"  Evaluated {combos_evaluated} combinations")
    print(f"  Found {len(all_trades)} favorable fair trades")

    if not all_trades:
        print("No favorable fair trades. Consider adjusting parameters.")

    return all_trades


# =============================================================================
# VERIFICATION
# =============================================================================


def verify_trade_impact(
    send_players: list[str],
    receive_players: list[str],
    my_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict:
    """
    Verify a trade's impact by recomputing win probability from scratch.
    """
    # Assertions
    send_set = set(send_players)
    receive_set = set(receive_players)

    assert send_set <= my_roster_names, (
        f"Cannot send players not on roster: {send_set - my_roster_names}"
    )
    assert not (receive_set & my_roster_names), (
        f"Already have some receive players: {receive_set & my_roster_names}"
    )

    # Compute before
    old_totals = compute_team_totals(my_roster_names, projections)
    V_before, diag_before = compute_win_probability(
        old_totals, opponent_totals, category_sigmas
    )

    # Compute after
    new_roster = (my_roster_names - send_set) | receive_set
    new_totals = compute_team_totals(new_roster, projections)
    V_after, diag_after = compute_win_probability(
        new_totals, opponent_totals, category_sigmas
    )

    # Category changes
    category_changes = {
        cat: new_totals[cat] - old_totals[cat] for cat in ALL_CATEGORIES
    }

    # Find matchup flips
    matchup_flips = []
    for cat in ALL_CATEGORIES:
        for opp_id in opponent_totals.keys():
            prob_before = diag_before["matchup_probs"][cat][opp_id]
            prob_after = diag_after["matchup_probs"][cat][opp_id]

            win_before = prob_before > 0.5
            win_after = prob_after > 0.5

            if win_before != win_after:
                matchup_flips.append(
                    {
                        "category": cat,
                        "opponent": opp_id,
                        "before": "win" if win_before else "lose",
                        "after": "win" if win_after else "lose",
                    }
                )

    print("Trade verification:")
    print(
        f"  Win probability: {V_before:.1%} → {V_after:.1%} ({V_after - V_before:+.1%})"
    )
    print(
        f"  Expected wins: {diag_before['expected_wins']:.1f} → {diag_after['expected_wins']:.1f}"
    )
    print(f"  Matchups flipped: {len(matchup_flips)}")

    return {
        "V_before": V_before,
        "V_after": V_after,
        "ewa": diag_after["expected_wins"] - diag_before["expected_wins"],
        "old_totals": old_totals,
        "new_totals": new_totals,
        "category_changes": category_changes,
        "ew_before": diag_before["expected_wins"],
        "ew_after": diag_after["expected_wins"],
        "matchup_flips": matchup_flips,
    }


# =============================================================================
# REPORTING
# =============================================================================


def compute_roster_situation(
    my_roster_names: set[str],
    projections: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> dict:
    """
    Compute full roster situation analysis. Main entry point.
    """
    my_totals = compute_team_totals(my_roster_names, projections)
    category_sigmas = estimate_projection_uncertainty(my_totals, opponent_totals)
    V, diagnostics = compute_win_probability(
        my_totals, opponent_totals, category_sigmas
    )

    # Analyze matchup probabilities
    matchup_probs = diagnostics["matchup_probs"]

    # Identify strengths and weaknesses
    strengths = []
    weaknesses = []

    for cat in ALL_CATEGORIES:
        avg_prob = np.mean([matchup_probs[cat][o] for o in opponent_totals.keys()])
        if avg_prob > 0.7:
            strengths.append(cat)
        elif avg_prob < 0.3:
            weaknesses.append(cat)

    # Category analysis
    category_analysis = {}
    for cat in ALL_CATEGORIES:
        avg_opp = np.mean([opponent_totals[o][cat] for o in opponent_totals.keys()])
        avg_prob = np.mean([matchup_probs[cat][o] for o in opponent_totals.keys()])

        if avg_prob > 0.7:
            status = "Strong"
        elif avg_prob > 0.5:
            status = "Moderate"
        elif avg_prob > 0.3:
            status = "Contested"
        else:
            status = "Weak"

        category_analysis[cat] = {
            "my_value": my_totals[cat],
            "avg_opponent": avg_opp,
            "avg_win_prob": avg_prob,
            "status": status,
        }

    return {
        "my_totals": my_totals,
        "category_sigmas": category_sigmas,
        "win_probability": V,
        "diagnostics": diagnostics,
        "expected_wins": diagnostics["expected_wins"],
        "expected_roto_points": diagnostics["expected_roto_points"],
        "category_analysis": category_analysis,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


def print_trade_report(
    situation: dict,
    trade_candidates: list[dict],
    player_values: pd.DataFrame,
    top_n: int = 5,
) -> None:
    """
    Print a formatted trade recommendation report.
    """
    print("\n" + "=" * 70)
    print("ROSTER SITUATION")
    print("=" * 70)

    print(f"\nWin probability: {situation['win_probability']:.1%}")
    print(
        f"Expected wins: {situation['expected_wins']:.1f}/60 | Expected roto points: {situation['expected_roto_points']}/70"
    )

    print("\nCATEGORY ANALYSIS:\n")
    print(
        f"{'Category':<10} {'My Value':>10} {'Avg Opp':>10} {'P(Win)':>8} {'Status':<10}"
    )
    print("-" * 50)

    for cat in ALL_CATEGORIES:
        analysis = situation["category_analysis"][cat]
        val = analysis["my_value"]
        avg_opp = analysis["avg_opponent"]

        if cat in ["ERA", "WHIP", "OPS"]:
            val_str = f"{val:.3f}"
            opp_str = f"{avg_opp:.3f}"
        else:
            val_str = f"{int(val)}"
            opp_str = f"{int(avg_opp)}"

        print(
            f"{cat:<10} {val_str:>10} {opp_str:>10} {analysis['avg_win_prob']:>7.0%} {analysis['status']:<10}"
        )

    print(
        f"\nSTRENGTHS: {', '.join(situation['strengths']) or 'None'} (high win probability)"
    )
    print(
        f"WEAKNESSES: {', '.join(situation['weaknesses']) or 'None'} (low win probability)"
    )

    print("\n" + "=" * 70)
    print("TOP TRADE RECOMMENDATIONS")
    print("=" * 70)

    if not trade_candidates:
        print("\nNo favorable fair trades found.")
        print("Consider adjusting parameters or accepting current roster.")
        return

    for i, trade in enumerate(trade_candidates[:top_n], 1):
        print(f"\n#{i}: {trade['ewa']:+.2f} expected wins")

        send_str = ", ".join(strip_name_suffix(p) for p in trade["send_players"])
        receive_str = ", ".join(strip_name_suffix(p) for p in trade["receive_players"])

        print(f"    Send:    {send_str}")
        print(
            f"    Receive: {receive_str} [from opponent {trade.get('trade_partner_id', '?')}]"
        )
        print("-" * 50)
        print(f"    Net: {trade['ewa']:+.2f} expected wins")
        print(
            f"    SGP: {trade['delta_generic']:+.1f} ({'Fair' if trade['is_fair'] else 'Unfair'})"
        )
        print(f"    Recommendation: {trade['recommendation']}")

    print("\n" + "=" * 70)
