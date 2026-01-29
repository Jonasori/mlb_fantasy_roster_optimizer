"""
Trade Engine using Probabilistic Win Model.

This module answers the question: "Which trades improve my chances of winning
the rotisserie league while being fair enough for opponents to accept?"

Based on the probabilistic rotisserie optimization framework from:
Rosenof, Z. (2025). "Optimizing for Rotisserie Fantasy Basketball." arXiv:2501.00933.

Key insight: A player's value is context-dependent. Production in a category
where you're safely ahead has near-zero marginal value. Production in a
contested race is extremely valuable.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from .data_loader import (
    ALL_CATEGORIES,
    MAX_HITTERS,
    MAX_PITCHERS,
    MIN_HITTERS,
    MIN_PITCHERS,
    MIN_STD,
    NEGATIVE_CATEGORIES,
    NUM_OPPONENTS,
    compute_team_totals,
    estimate_projection_uncertainty,
    strip_name_suffix,
)

# === TRADE ENGINE CONFIGURATION ===

# Expected value and variance of maximum of N standard normals
# From Teichroew (1956), used by Rosenof (2025)
MEV_TABLE = {1: 0.0, 2: 0.564, 3: 0.846, 4: 1.029, 5: 1.163, 6: 1.267}
MVAR_TABLE = {1: 1.0, 2: 0.682, 3: 0.559, 4: 0.492, 5: 0.448, 6: 0.416}

# Trade fairness threshold: percentage-based
# A fair trade has SGP differential within 10% of total SGP involved
# Example: 4.5 SGP for 6.2 SGP = 1.7 diff / 10.7 total = 16% -> UNFAIR
# Example: 3.0 SGP for 3.5 SGP = 0.5 diff / 6.5 total = 8% -> FAIR
FAIRNESS_THRESHOLD_PERCENT = 0.10  # 10% max differential

# Maximum players per side in a trade
MAX_TRADE_SIZE = 3


# === GAP AND PROBABILITY COMPUTATION ===


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
    """
    gaps = {}
    sqrt2 = np.sqrt(2)

    for category in ALL_CATEGORIES:
        gaps[category] = {}
        sigma = category_sigmas[category]

        for opp_id, opp_totals in opponent_totals.items():
            my_val = my_totals[category]
            opp_val = opp_totals[category]

            if category in NEGATIVE_CATEGORIES:
                # Lower is better, flip sign so positive = winning
                raw_gap = opp_val - my_val
            else:
                # Higher is better
                raw_gap = my_val - opp_val

            # Normalize by σ√2
            normalized = raw_gap / (sigma * sqrt2) if sigma > MIN_STD else 0.0
            gaps[category][opp_id] = normalized

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

    for category in ALL_CATEGORIES:
        probs[category] = {}
        for opp_id, gap in normalized_gaps[category].items():
            probs[category][opp_id] = stats.norm.cdf(gap)

    return probs


# === WIN PROBABILITY COMPUTATION ===


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

    Returns:
        V: Victory probability (0 to 1)
        diagnostics: Dict with intermediate values
    """
    n_opp = len(opponent_totals)
    n_cats = len(ALL_CATEGORIES)

    # Step 1: Compute normalized gaps
    normalized_gaps = compute_normalized_gaps(
        my_totals, opponent_totals, category_sigmas
    )

    # Step 2: Compute matchup probabilities
    matchup_probs = compute_matchup_probabilities(normalized_gaps)

    # Step 3: Expected fantasy points (μ_T)
    mu_T = sum(
        matchup_probs[c][o] for c in ALL_CATEGORIES for o in opponent_totals.keys()
    )

    # Step 4: Variance of fantasy points (σ_T²)
    # Base variance: sum of Φ(1-Φ) terms (Bernoulli variance)
    sigma_T_sq = sum(
        matchup_probs[c][o] * (1 - matchup_probs[c][o])
        for c in ALL_CATEGORIES
        for o in opponent_totals.keys()
    )

    # Note: Full correlation adjustment is complex; omitting for simplicity
    # This assumes independence between matchups, which underestimates variance

    # Step 5: Expected variance for opponent fantasy points (E[σ_M²])
    # Approximation: assume similar variance for all teams
    E_sigma_M_sq = sigma_T_sq  # Simplified assumption

    # Step 6: Target to beat (max of opponents)
    mev = MEV_TABLE.get(n_opp, 1.267)
    mvar = MVAR_TABLE.get(n_opp, 0.416)

    mu_L = mev * np.sqrt(E_sigma_M_sq) if E_sigma_M_sq > 0 else 0
    sigma_L_sq = E_sigma_M_sq * mvar

    # Step 7: Differential distribution
    # μ_D = μ_T * (|O|+1)/|O| - |C|*(|O|+1)/2 - μ_L
    scale_factor = (n_opp + 1) / n_opp
    mu_D = mu_T * scale_factor - n_cats * (n_opp + 1) / 2 - mu_L

    # σ_D² = ((|O|+1)/|O|) * σ_T² + σ_L²
    sigma_D_sq = scale_factor * sigma_T_sq + sigma_L_sq
    sigma_D = np.sqrt(sigma_D_sq) if sigma_D_sq > 0 else MIN_STD

    # Step 8: Victory probability
    V = stats.norm.cdf(mu_D / sigma_D) if sigma_D > MIN_STD else 0.5

    # Compute expected roto points (rough estimate)
    # For each category, expected rank ≈ 1 + n_opp * (1 - avg_win_prob)
    # Roto points per category = 8 - rank = 7 - n_opp * (1 - avg_win_prob)
    expected_roto_points = 0.0
    for cat in ALL_CATEGORIES:
        avg_win_prob = np.mean(list(matchup_probs[cat].values()))
        expected_rank = 1 + n_opp * (1 - avg_win_prob)
        expected_roto_points += 8 - expected_rank

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


# === MARGINAL VALUE COMPUTATION ===


def compute_win_probability_gradient(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict[str, dict[int, float]]:
    """
    Compute the gradient of win probability with respect to each matchup gap.

    Returns:
        Dict[category, Dict[opponent_id, gradient]]
    """
    V, diag = compute_win_probability(my_totals, opponent_totals, category_sigmas)

    sigma_D = diag["sigma_D"]
    mu_D = diag["mu_D"]
    n_opp = len(opponent_totals)
    scale_factor = (n_opp + 1) / n_opp

    # Gradient of V with respect to μ_D
    dV_dmuD = stats.norm.pdf(mu_D / sigma_D) / sigma_D if sigma_D > MIN_STD else 0

    gradient = {}
    normalized_gaps = diag["normalized_gaps"]

    for category in ALL_CATEGORIES:
        gradient[category] = {}
        for opp_id in opponent_totals.keys():
            gap = normalized_gaps[category][opp_id]
            # ∇_{c,o}(μ_T) = φ(μ_c,o) (PDF of standard normal at gap)
            # ∇_{c,o}(μ_D) ≈ scale_factor * φ(μ_c,o)
            d_mu_T = stats.norm.pdf(gap)
            d_mu_D = scale_factor * d_mu_T
            gradient[category][opp_id] = dV_dmuD * d_mu_D

    return gradient


def _compute_totals_with_player_change(
    player_name: str,
    projections: pd.DataFrame,
    roster_names: set[str],
    is_adding: bool,
) -> dict[str, float]:
    """
    Compute new totals after adding or removing a player.

    Simply modifies the roster set and calls compute_team_totals.
    """
    if is_adding:
        new_roster = roster_names | {player_name}
    else:
        new_roster = roster_names - {player_name}

    return compute_team_totals(new_roster, projections)


def compute_player_marginal_value(
    player_name: str,
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    my_roster_names: set[str],
    is_acquisition: bool = True,
) -> tuple[float, dict]:
    """
    Compute the marginal change in win probability from acquiring/losing a player.

    Uses numerical differentiation: computes V before and after.

    Returns:
        delta_V: Change in win probability (can be negative)
        breakdown: Dict with per-category contributions
    """
    assert player_name in projections["Name"].values, f"Player not found: {player_name}"

    # Compute V before
    V_before, _ = compute_win_probability(my_totals, opponent_totals, category_sigmas)

    # Compute new totals
    new_totals = _compute_totals_with_player_change(
        player_name, projections, my_roster_names, is_adding=is_acquisition
    )

    # Compute V after
    V_after, _ = compute_win_probability(new_totals, opponent_totals, category_sigmas)

    delta_V = V_after - V_before

    # Compute per-category breakdown (simplified)
    breakdown = {cat: new_totals[cat] - my_totals[cat] for cat in ALL_CATEGORIES}

    return delta_V, breakdown


# === PLAYER VALUE DATAFRAME ===


def compute_player_values(
    player_names: set[str],
    my_roster_names: set[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> pd.DataFrame:
    """
    Compute marginal value for a set of players.

    Returns:
        DataFrame with columns:
            Name, player_type, Position, on_my_roster,
            delta_V_acquire, delta_V_lose, generic_value (SGP)
    """
    records = []

    # Use SGP (Standing Gain Points) as generic value - a context-free
    # player valuation metric computed from our league's scoring categories
    assert "SGP" in projections.columns, (
        "SGP column not found in projections. "
        "Make sure load_projections() is called to compute SGP values."
    )
    generic_values = projections[["Name", "SGP"]].copy()
    generic_values = generic_values.rename(columns={"SGP": "generic_value"})

    for name in tqdm(player_names, desc="Computing player values"):
        player_row = projections[projections["Name"] == name]
        if len(player_row) == 0:
            continue

        player = player_row.iloc[0]
        on_my_roster = name in my_roster_names

        # Generic value
        gv_row = generic_values[generic_values["Name"] == name]
        generic_val = gv_row["generic_value"].iloc[0] if len(gv_row) > 0 else 0.0

        # Delta V for acquisition
        if on_my_roster:
            delta_V_acquire = 0.0  # Already have them
        else:
            delta_V_acquire, _ = compute_player_marginal_value(
                name,
                projections,
                my_totals,
                opponent_totals,
                category_sigmas,
                my_roster_names,
                is_acquisition=True,
            )

        # Delta V for losing
        if on_my_roster:
            delta_V_lose, _ = compute_player_marginal_value(
                name,
                projections,
                my_totals,
                opponent_totals,
                category_sigmas,
                my_roster_names,
                is_acquisition=False,
            )
            delta_V_lose = -delta_V_lose  # Make positive = cost of losing
        else:
            delta_V_lose = np.nan

        records.append(
            {
                "Name": name,
                "player_type": player["player_type"],
                "Position": player["Position"],
                "on_my_roster": on_my_roster,
                "delta_V_acquire": delta_V_acquire,
                "delta_V_lose": delta_V_lose,
                "generic_value": generic_val,
            }
        )

    result = pd.DataFrame(records)
    return result.sort_values("delta_V_acquire", ascending=False)


# === TRADE CANDIDATE IDENTIFICATION ===


def identify_trade_targets(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    opponent_rosters: dict[int, set[str]],
    n_targets: int = 15,
) -> pd.DataFrame:
    """
    Identify players to TARGET (acquire) in trades.

    Targets are players on opponent rosters who are valuable TO ME but not
    universally valuable (i.e., hidden gems, not superstars everyone wants).

    IMPORTANT: Ensures a mix of hitters and pitchers in targets so that
    roster composition can be fixed via trades. If roster needs pitchers,
    at least 40% of targets will be pitchers (and vice versa for hitters).
    """
    # Filter to opponent players
    all_opp_names = set().union(*opponent_rosters.values())
    targets = player_values[player_values["Name"].isin(all_opp_names)].copy()

    # Add owner_id
    def find_owner(name):
        for opp_id, roster in opponent_rosters.items():
            if name in roster:
                return opp_id
        return None

    targets["owner_id"] = targets["Name"].apply(find_owner)

    # Compute "acquirability score": ratio of value-to-me vs market-value
    targets["acquirability"] = targets["delta_V_acquire"] / (
        targets["generic_value"].clip(lower=0.5) + 0.5
    )

    # Filter to players with positive value to me
    targets = targets[targets["delta_V_acquire"] > 0]

    # CRITICAL: Ensure mix of hitters and pitchers for roster composition flexibility
    # Take top targets from each player type separately, then combine
    hitter_targets = targets[targets["player_type"] == "hitter"]
    pitcher_targets = targets[targets["player_type"] == "pitcher"]

    # Minimum 40% of each type (so roster composition can be fixed)
    min_per_type = max(1, n_targets * 2 // 5)  # At least 40%
    remaining = n_targets - 2 * min_per_type

    # Take minimum from each, then fill remaining with best overall
    top_hitters = hitter_targets.nlargest(min_per_type + remaining, "acquirability")
    top_pitchers = pitcher_targets.nlargest(min_per_type + remaining, "acquirability")

    # Combine: take min_per_type from each, then fill remaining with best overall
    result = pd.concat(
        [
            top_hitters.head(min_per_type),
            top_pitchers.head(min_per_type),
        ]
    )

    # Add remaining from the better pool
    remaining_hitters = top_hitters.iloc[min_per_type:]
    remaining_pitchers = top_pitchers.iloc[min_per_type:]
    remaining_combined = pd.concat([remaining_hitters, remaining_pitchers])
    remaining_combined = remaining_combined.nlargest(remaining, "acquirability")

    targets = pd.concat([result, remaining_combined])
    targets = targets.head(n_targets)

    if len(targets) > 0:
        n_h = (targets["player_type"] == "hitter").sum()
        n_p = (targets["player_type"] == "pitcher").sum()
        best = targets.iloc[0]
        print(
            f"Trade targets: {len(targets)} players identified ({n_h} hitters, {n_p} pitchers)"
        )
        print(
            f"  Best: {strip_name_suffix(best['Name'])} from Team {best['owner_id']} "
            f"(value to me: +{best['delta_V_acquire']:.3f} W, SGP: {best['generic_value']:.1f})"
        )

    return targets


def identify_trade_pieces(
    player_values: pd.DataFrame,
    my_roster_names: set[str],
    n_pieces: int = 15,
) -> pd.DataFrame:
    """
    Identify players to OFFER (trade away) from my roster.

    Expendability formula (single formula, no conditionals):
        expendability = -(WAR + lose_cost * scale)

    This naturally handles both normal and edge cases:
    - Low SGP → more expendable (not a valuable asset)
    - Low lose_cost → more expendable (doesn't hurt to lose)
    - When win prob is at floor/ceiling and lose_cost ≈ 0 for everyone,
      the formula degrades gracefully to ranking by -SGP

    IMPORTANT: Ensures a mix of hitters and pitchers so that roster composition
    can be fixed via trades.
    """
    pieces = player_values[player_values["on_my_roster"]].copy()

    # Single formula: low SGP and low lose_cost = high expendability
    # Scale factor converts lose_cost (0.0001-0.01) to SGP scale (0-20)
    LOSE_COST_SCALE = 200  # 0.01 lose_cost → 2.0 SGP-equivalent
    pieces["expendability_score"] = -(
        pieces["generic_value"].fillna(0)
        + pieces["delta_V_lose"].fillna(0) * LOSE_COST_SCALE
    )

    # CRITICAL: Ensure mix of hitters and pitchers for roster composition flexibility
    hitter_pieces = pieces[pieces["player_type"] == "hitter"]
    pitcher_pieces = pieces[pieces["player_type"] == "pitcher"]

    # Minimum 40% of each type
    min_per_type = max(1, n_pieces * 2 // 5)
    remaining = n_pieces - 2 * min_per_type

    # Take most expendable from each type
    top_hitters = hitter_pieces.nlargest(
        min_per_type + remaining, "expendability_score"
    )
    top_pitchers = pitcher_pieces.nlargest(
        min_per_type + remaining, "expendability_score"
    )

    result = pd.concat(
        [
            top_hitters.head(min_per_type),
            top_pitchers.head(min_per_type),
        ]
    )

    remaining_hitters = top_hitters.iloc[min_per_type:]
    remaining_pitchers = top_pitchers.iloc[min_per_type:]
    remaining_combined = pd.concat([remaining_hitters, remaining_pitchers])
    remaining_combined = remaining_combined.nlargest(remaining, "expendability_score")

    pieces = pd.concat([result, remaining_combined])
    pieces = pieces.head(n_pieces)

    if len(pieces) > 0:
        n_h = (pieces["player_type"] == "hitter").sum()
        n_p = (pieces["player_type"] == "pitcher").sum()
        best = pieces.iloc[0]
        print(f"Trade pieces: {len(pieces)} players ({n_h} hitters, {n_p} pitchers)")
        print(
            f"  Most expendable: {strip_name_suffix(best['Name'])} "
            f"(SGP={best['generic_value']:.1f}, lose_cost={best['delta_V_lose']:.3f})"
        )

    return pieces


# === TRADE EVALUATION ===


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
) -> dict:
    """
    Evaluate a specific trade proposal.

    Returns:
        Dict with trade analysis including delta_V, fairness, recommendation.
    """
    # Validate inputs
    for p in send_players:
        assert p in my_roster_names, f"Cannot send {p} — not on my roster"
    for p in receive_players:
        assert p not in my_roster_names, f"Already have {p} on roster"
        assert p in projections["Name"].values, f"Player not found: {p}"

    assert len(set(send_players) & set(receive_players)) == 0, (
        "Cannot trade player with themselves"
    )

    # Compute V_before
    V_before, _ = compute_win_probability(my_totals, opponent_totals, category_sigmas)

    # Compute new roster and totals
    new_roster_names = (my_roster_names - set(send_players)) | set(receive_players)
    new_totals = compute_team_totals(new_roster_names, projections)

    # Validate composition
    new_roster_df = projections[projections["Name"].isin(new_roster_names)]
    n_hitters = (new_roster_df["player_type"] == "hitter").sum()
    n_pitchers = (new_roster_df["player_type"] == "pitcher").sum()

    assert MIN_HITTERS <= n_hitters <= MAX_HITTERS, (
        f"Trade violates hitter bounds: {n_hitters} (need {MIN_HITTERS}-{MAX_HITTERS})"
    )
    assert MIN_PITCHERS <= n_pitchers <= MAX_PITCHERS, (
        f"Trade violates pitcher bounds: {n_pitchers} (need {MIN_PITCHERS}-{MAX_PITCHERS})"
    )

    # Compute V_after
    V_after, _ = compute_win_probability(new_totals, opponent_totals, category_sigmas)

    delta_V = V_after - V_before

    # Compute generic value change
    send_generic = sum(
        player_values[player_values["Name"] == p]["generic_value"].iloc[0]
        for p in send_players
        if p in player_values["Name"].values
    )
    receive_generic = sum(
        player_values[player_values["Name"] == p]["generic_value"].iloc[0]
        for p in receive_players
        if p in player_values["Name"].values
    )
    delta_generic = receive_generic - send_generic

    # Fairness: SGP differential should be within threshold of total SGP
    # This prevents "steals" where one side clearly wins on market value
    total_war = send_generic + receive_generic
    if total_war > 0:
        relative_diff = abs(delta_generic) / total_war
        is_fair = relative_diff <= FAIRNESS_THRESHOLD_PERCENT
    else:
        is_fair = True  # Edge case: no SGP involved

    # Require meaningful improvement (at least 0.1% = 0.001 win probability)
    # to avoid recommending trades with negligible impact
    MIN_MEANINGFUL_IMPROVEMENT = 0.001
    is_good_for_me = delta_V >= MIN_MEANINGFUL_IMPROVEMENT
    is_bad_for_me = delta_V <= -MIN_MEANINGFUL_IMPROVEMENT
    is_neutral = not is_good_for_me and not is_bad_for_me

    # Recommendation
    if is_good_for_me and is_fair:
        recommendation = "ACCEPT"
    elif is_good_for_me and not is_fair and delta_generic > 0:
        recommendation = "STEAL"  # I'm getting more generic value
    elif is_bad_for_me and is_fair:
        recommendation = "REJECT"
    elif is_neutral and is_fair:
        recommendation = "NEUTRAL"  # Fair trade but negligible impact
    else:
        recommendation = "UNFAIR"

    # Category impact
    category_impact = {cat: new_totals[cat] - my_totals[cat] for cat in ALL_CATEGORIES}

    # Get individual player generic values for display
    send_generics = []
    for p in send_players:
        pv_row = player_values[player_values["Name"] == p]
        if len(pv_row) > 0:
            gv = pv_row["generic_value"].iloc[0]
            send_generics.append((p, gv if pd.notna(gv) else 0.0))
        else:
            send_generics.append((p, 0.0))

    receive_generics = []
    for p in receive_players:
        pv_row = player_values[player_values["Name"] == p]
        if len(pv_row) > 0:
            gv = pv_row["generic_value"].iloc[0]
            receive_generics.append((p, gv if pd.notna(gv) else 0.0))
        else:
            receive_generics.append((p, 0.0))

    # Find which opponent we're trading with
    trade_partner_id = None
    if opponent_rosters is not None:
        for p in receive_players:
            for opp_id, roster in opponent_rosters.items():
                if p in roster:
                    trade_partner_id = opp_id
                    break
            if trade_partner_id is not None:
                break

    result = {
        "send_players": send_players,
        "receive_players": receive_players,
        "delta_V": delta_V,
        "delta_generic": delta_generic,
        "V_before": V_before,
        "V_after": V_after,
        "is_fair": is_fair,
        "is_good_for_me": is_good_for_me,
        "recommendation": recommendation,
        "category_impact": category_impact,
        "send_generics": send_generics,
        "receive_generics": receive_generics,
        "trade_partner_id": trade_partner_id,
    }

    # Print summary only for fair trades
    if is_fair:
        send_str = ", ".join(
            f"{strip_name_suffix(p)} (SGP: {v:.1f})" for p, v in send_generics
        )
        receive_str = ", ".join(
            f"{strip_name_suffix(p)} (SGP: {v:.1f})" for p, v in receive_generics
        )
        partner_str = f" with Team {trade_partner_id}" if trade_partner_id else ""
        print(f"Trade{partner_str}:")
        print(f"  Send: [{send_str}]")
        print(f"  Receive: [{receive_str}]")
        print(f"  Net to me: {delta_V:+.3f} win probability ({delta_V * 100:+.1f}%)")
        print(f"  Recommendation: {recommendation}")

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
) -> list[dict]:
    """
    Generate candidate trades to consider.

    Returns:
        List of trade evaluation dicts, sorted by delta_V descending.
        Only includes trades where is_fair=True and is_good_for_me=True.
    """
    # Get targets and pieces
    targets = identify_trade_targets(
        player_values, my_roster_names, opponent_rosters, n_targets
    )
    pieces = identify_trade_pieces(player_values, my_roster_names, n_pieces)

    if len(targets) == 0 or len(pieces) == 0:
        print("No favorable fair trades. Consider adjusting parameters.")
        return []

    # Generate combinations
    target_names = list(targets["Name"])
    piece_names = list(pieces["Name"])

    # Group targets by owner
    targets_by_owner = {}
    for _, row in targets.iterrows():
        owner = row["owner_id"]
        if owner not in targets_by_owner:
            targets_by_owner[owner] = []
        targets_by_owner[owner].append(row["Name"])

    candidates = []
    evaluated = 0
    skipped_composition = 0

    # Get current roster composition
    my_roster_df = projections[projections["Name"].isin(my_roster_names)]
    current_hitters = (my_roster_df["player_type"] == "hitter").sum()
    current_pitchers = (my_roster_df["player_type"] == "pitcher").sum()

    # Helper to check if trade maintains valid composition
    def _would_violate_composition(
        send_names: list[str], receive_names: list[str]
    ) -> bool:
        send_df = projections[projections["Name"].isin(send_names)]
        receive_df = projections[projections["Name"].isin(receive_names)]

        send_hitters = (send_df["player_type"] == "hitter").sum()
        send_pitchers = (send_df["player_type"] == "pitcher").sum()
        receive_hitters = (receive_df["player_type"] == "hitter").sum()
        receive_pitchers = (receive_df["player_type"] == "pitcher").sum()

        new_hitters = current_hitters - send_hitters + receive_hitters
        new_pitchers = current_pitchers - send_pitchers + receive_pitchers

        if not (MIN_HITTERS <= new_hitters <= MAX_HITTERS):
            return True
        if not (MIN_PITCHERS <= new_pitchers <= MAX_PITCHERS):
            return True
        return False

    # Generate 1-for-1, 2-for-1, 1-for-2, 2-for-2 combinations
    for send_size in range(1, min(max_send, len(piece_names)) + 1):
        for receive_size in range(1, min(max_receive, len(target_names)) + 1):
            for send_combo in combinations(piece_names, send_size):
                for owner, owner_targets in targets_by_owner.items():
                    for receive_combo in combinations(owner_targets, receive_size):
                        send_list = list(send_combo)
                        receive_list = list(receive_combo)

                        # Pre-filter: skip trades that violate roster composition
                        if _would_violate_composition(send_list, receive_list):
                            skipped_composition += 1
                            continue

                        evaluated += 1

                        # Evaluate trade
                        result = evaluate_trade(
                            send_list,
                            receive_list,
                            player_values,
                            my_roster_names,
                            projections,
                            my_totals,
                            opponent_totals,
                            category_sigmas,
                            opponent_rosters,
                        )

                        if result["is_fair"] and result["is_good_for_me"]:
                            candidates.append(result)

    # Sort by delta_V
    candidates.sort(key=lambda x: x["delta_V"], reverse=True)
    candidates = candidates[:n_candidates]

    print(f"Generated {len(candidates)} candidate trades")
    print(
        f"  Evaluated {evaluated} combinations (skipped {skipped_composition} for composition)"
    )
    print(f"  Found {len(candidates)} favorable fair trades")

    if len(candidates) == 0:
        print("No favorable fair trades. Consider adjusting parameters.")

    return candidates


# === VERIFICATION ===


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

    This is the ground truth check — full recomputation, not gradient-based.
    """
    # Validate inputs
    for p in send_players:
        assert p in my_roster_names, f"Cannot send {p} — not on my roster"
    for p in receive_players:
        assert p not in my_roster_names, f"Already have {p} on roster"

    # Compute before
    old_totals = compute_team_totals(my_roster_names, projections)
    V_before, diag_before = compute_win_probability(
        old_totals, opponent_totals, category_sigmas
    )

    # Compute after
    new_roster = (my_roster_names - set(send_players)) | set(receive_players)
    new_totals = compute_team_totals(new_roster, projections)
    V_after, diag_after = compute_win_probability(
        new_totals, opponent_totals, category_sigmas
    )

    # Category changes
    category_changes = {
        cat: new_totals[cat] - old_totals[cat] for cat in ALL_CATEGORIES
    }

    # Matchup flips
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
                        "prob_before": prob_before,
                        "prob_after": prob_after,
                    }
                )

    result = {
        "V_before": V_before,
        "V_after": V_after,
        "delta_V": V_after - V_before,
        "old_totals": old_totals,
        "new_totals": new_totals,
        "category_changes": category_changes,
        "wins_before": diag_before["expected_wins"],
        "wins_after": diag_after["expected_wins"],
        "matchup_flips": matchup_flips,
    }

    # Print summary
    print("Trade verification:")
    print(
        f"  Win probability: {V_before:.1%} → {V_after:.1%} ({V_after - V_before:+.1%})"
    )
    print(
        f"  Expected wins: {diag_before['expected_wins']:.1f} → {diag_after['expected_wins']:.1f}"
    )
    print(f"  Matchups flipped: {len(matchup_flips)}")

    return result


# === REPORTING ===


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

    # Analyze strengths and weaknesses
    matchup_probs = diagnostics["matchup_probs"]

    strengths = []
    weaknesses = []

    for cat in ALL_CATEGORIES:
        avg_prob = np.mean(list(matchup_probs[cat].values()))
        if avg_prob > 0.7:
            strengths.append(cat)
        elif avg_prob < 0.3:
            weaknesses.append(cat)

    # Category analysis
    category_analysis = {}
    for cat in ALL_CATEGORIES:
        opp_avg = np.mean([opponent_totals[o][cat] for o in opponent_totals])
        avg_prob = np.mean(list(matchup_probs[cat].values()))

        if avg_prob > 0.7:
            status = "Strong"
        elif avg_prob > 0.5:
            status = "Ahead"
        elif avg_prob > 0.3:
            status = "Contested"
        else:
            status = "Weak"

        category_analysis[cat] = {
            "my_value": my_totals[cat],
            "opponent_avg": opp_avg,
            "win_probability": avg_prob,
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

    print(f"Win probability: {situation['win_probability']:.1%}")
    print(f"Expected wins: {situation['expected_wins']:.1f}/60")

    print("\nCATEGORY ANALYSIS:\n")
    print(
        f"{'Category':<10} {'My Value':<12} {'Avg Opp':<12} {'P(Win)':<10} {'Status':<10}"
    )
    print("-" * 54)

    for cat in ALL_CATEGORIES:
        analysis = situation["category_analysis"][cat]
        my_val = analysis["my_value"]
        opp_avg = analysis["opponent_avg"]
        prob = analysis["win_probability"]
        status = analysis["status"]

        if cat in ["ERA", "WHIP", "OPS"]:
            val_fmt = f"{my_val:.3f}"
            opp_fmt = f"{opp_avg:.3f}"
        else:
            val_fmt = f"{my_val:.0f}"
            opp_fmt = f"{opp_avg:.0f}"

        print(
            f"{cat:<10} {val_fmt:<12} {opp_fmt:<12} {prob * 100:>6.0f}%    {status:<10}"
        )

    if situation["strengths"]:
        print(f"\nSTRENGTHS: {', '.join(situation['strengths'])}")
    if situation["weaknesses"]:
        print(f"WEAKNESSES: {', '.join(situation['weaknesses'])}")

    print("\n" + "=" * 70)
    print("TOP TRADE RECOMMENDATIONS")
    print("=" * 70)

    if not trade_candidates:
        print("\nNo favorable fair trades found.")
        print("Consider:")
        print("  - Increasing n_targets or n_pieces")
        print("  - Relaxing fairness threshold")
        print("  - Looking at larger trades (max_send, max_receive)")
        return

    for i, trade in enumerate(trade_candidates[:top_n], 1):
        partner_str = (
            f" with Team {trade['trade_partner_id']}"
            if trade.get("trade_partner_id")
            else ""
        )
        print(f"\n#{i}{partner_str}: {trade['delta_V'] * 100:+.1f}% win probability")

        # Format send players with their generic values
        if "send_generics" in trade:
            send_str = ", ".join(
                f"{strip_name_suffix(p)} (SGP: {v:.1f})"
                for p, v in trade["send_generics"]
            )
        else:
            send_str = ", ".join(strip_name_suffix(p) for p in trade["send_players"])

        # Format receive players with their generic values
        if "receive_generics" in trade:
            receive_str = ", ".join(
                f"{strip_name_suffix(p)} (SGP: {v:.1f})"
                for p, v in trade["receive_generics"]
            )
        else:
            receive_str = ", ".join(
                strip_name_suffix(p) for p in trade["receive_players"]
            )

        print(f"    Send:    [{send_str}]")
        print(f"    Receive: [{receive_str}]")
        print(f"    " + "-" * 50)
        print(f"    Net to me: {trade['delta_V'] * 100:+.1f}% win probability")
        print(
            f"    SGP change: {trade['delta_generic']:+.1f} ({'Fair' if trade['is_fair'] else 'Unfair'})"
        )
        print(f"    Recommendation: {trade['recommendation']}")

    print("\n" + "=" * 70)
