"""
Probabilistic win model and league position estimation.

EW = Σ_c Σ_o Φ(z_{c,o}): sum of pairwise beat probabilities.
Expected standing points = 10 + EW.

Depends only on config.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    ALL_CATEGORIES,
    MIN_STAT_STANDARD_DEVIATION,
    NEGATIVE_CATEGORIES,
)

# ============================================================================
# CONSTANTS — Projection noise floors
# ============================================================================

# Counting stats: σ = |league_mean| × CV
_TEAM_PROJECTION_CV: dict[str, float] = {
    "R": 0.06,
    "HR": 0.09,
    "RBI": 0.06,
    "SB": 0.15,
    "W": 0.12,
    "SV": 0.20,
    "K": 0.06,
}
# Ratio stats: absolute σ
_TEAM_PROJECTION_SIGMA: dict[str, float] = {
    "OPS": 0.012,
    "ERA": 0.30,
    "WHIP": 0.050,
}

# ============================================================================
# WIN PROBABILITY — EW computation
# ============================================================================


def compute_win_probability(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> tuple[float, dict]:
    """Compute expected wins via pairwise beat probabilities.

    EW = Σ_c Σ_o Φ(z_{c,o}): sum of pairwise beat probabilities.
    Expected standing points = 10 + EW.

    For category c against opponent o, the normalized gap:
        z_{c,o} = (my_c − opp_{c,o}) / (σ_c √2)  if c ∈ C⁺
        z_{c,o} = (opp_{c,o} − my_c) / (σ_c √2)  if c ∈ C⁻

    In both cases, positive z means I'm winning.

    Args:
        my_totals: My team's category totals (10 categories).
        opponent_totals: {opp_id: {cat: total}} for each opponent.
            Keys are 1-indexed opponent IDs.
        category_sigmas: σ_c per category (projection uncertainty).

    Returns:
        ew: Expected wins (float, range 0–60 for 10 categories × 6 opponents).
        diagnostics: Dict including:
            'expected_wins' (same as ew, for convenience),
            'beat_probs' (per-category per-opponent beat probabilities),
            'normalized_gaps' (z-scores).
    """
    normalized_gaps: dict[str, dict[int, float]] = {}
    beat_probs: dict[str, dict[int, float]] = {}

    for cat in ALL_CATEGORIES:
        normalized_gaps[cat] = {}
        beat_probs[cat] = {}
        sigma = max(category_sigmas[cat], MIN_STAT_STANDARD_DEVIATION)
        denom = sigma * np.sqrt(2)

        for opp_id, opp_totals in opponent_totals.items():
            if cat in NEGATIVE_CATEGORIES:
                z = (opp_totals[cat] - my_totals[cat]) / denom
            else:
                z = (my_totals[cat] - opp_totals[cat]) / denom
            normalized_gaps[cat][opp_id] = z
            beat_probs[cat][opp_id] = float(stats.norm.cdf(z))

    ew = sum(
        beat_probs[cat][opp_id] for cat in ALL_CATEGORIES for opp_id in opponent_totals
    )

    diagnostics = {
        "expected_wins": ew,
        "beat_probs": beat_probs,
        "normalized_gaps": normalized_gaps,
    }

    return ew, diagnostics


# ============================================================================
# EW GRADIENT
# ============================================================================


def compute_ew_gradient(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
) -> dict[str, float]:
    """∂EW/∂(my_c) for each category c.

    For c ∈ C⁺ (R, HR, RBI, SB, OPS, W, SV, K):
        g_c = +Σ_o φ(z_{c,o}) / (σ_c √2) > 0
    For c ∈ C⁻ (ERA, WHIP):
        g_c = −Σ_o φ(z_{c,o}) / (σ_c √2) < 0

    Magnitude reflects matchup sensitivity — large when matchups are close,
    small when matchups are already decided.
    """
    gradient: dict[str, float] = {}

    for cat in ALL_CATEGORIES:
        sigma = max(category_sigmas[cat], MIN_STAT_STANDARD_DEVIATION)
        denom = sigma * np.sqrt(2)
        total = 0.0

        for opp_totals in opponent_totals.values():
            if cat in NEGATIVE_CATEGORIES:
                z = (opp_totals[cat] - my_totals[cat]) / denom
            else:
                z = (my_totals[cat] - opp_totals[cat]) / denom
            total += stats.norm.pdf(z)

        sign = -1.0 if cat in NEGATIVE_CATEGORIES else 1.0
        gradient[cat] = sign * total / denom

    return gradient


# ============================================================================
# PROJECTION UNCERTAINTY
# ============================================================================


def estimate_projection_uncertainty(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    season_fraction_remaining: float | None = None,
) -> dict[str, float]:
    """Estimate σ_c: how much actual SEASON outcomes could deviate from projection.

    σ_c is the uncertainty in the standings gap for category c. It enters the
    win model as z = (gap) / (σ_c √2). Only the UNKNOWN portion of a season
    total is uncertain — banked YTD stats are already realized.

    Counting stats: σ_c = |league_mean_c| × CV_c (fixed CV per category).
    Ratio stats: σ_c = fixed absolute value per category.

    The CV / absolute constants are calibrated to FULL-SEASON dispersion. When
    ``season_fraction_remaining`` (f) is given, σ is rescaled to the remaining
    horizon. The standard result (verified by Monte Carlo) is that the SD of a
    season total — banked known plus ros projected — equals σ_full · √f for
    BOTH counting and ratio stats:

      • Counting σ is computed here from the ros league mean, which already
        carries one factor of f (ros_mean ≈ f · full_mean), giving σ_full · f.
        To reach σ_full · √f we divide by √f.
      • Ratio σ is a bare full-season constant (σ_full). To reach σ_full · √f
        we multiply by √f.

    At f = 1 (preseason) both reduce to the original full-season values, so
    behavior is unchanged when no fraction is supplied.

    IMPORTANT: pass ROS (not season) totals here — the league mean must be the
    rest-of-season magnitude for the counting-stat scaling to hold. Supply f
    ONLY when the totals reflect a banked+ros season model (see
    compute_league_state); leaving it None keeps the legacy ros-only behavior.

    CRITICAL: This is projection uncertainty, NOT observed cross-team
    variance. See W7 for why the distinction matters.

    Args:
        my_totals: My team's ROS category totals.
        opponent_totals: Dict mapping opponent_id to their ROS category totals.
        season_fraction_remaining: Fraction of the season still to be played,
            in (0, 1]. None disables horizon rescaling (legacy behavior).

    Returns:
        Dict mapping category to standard deviation.
    """
    if season_fraction_remaining is not None:
        assert 0.0 < season_fraction_remaining <= 1.0, (
            f"estimate_projection_uncertainty: season_fraction_remaining must be "
            f"in (0, 1], got {season_fraction_remaining}."
        )
        sqrt_f = season_fraction_remaining**0.5
        counting_scale = 1.0 / sqrt_f
        ratio_scale = sqrt_f
    else:
        counting_scale = 1.0
        ratio_scale = 1.0

    category_sigmas: dict[str, float] = {}

    for category in ALL_CATEGORIES:
        if category in _TEAM_PROJECTION_SIGMA:
            sigma = _TEAM_PROJECTION_SIGMA[category] * ratio_scale
        else:
            values = [my_totals[category]]
            for opp_id in sorted(opponent_totals.keys()):
                values.append(opponent_totals[opp_id][category])
            league_mean = abs(float(np.mean(values)))
            cv = _TEAM_PROJECTION_CV.get(category, 0.10)
            sigma = league_mean * cv * counting_scale

        category_sigmas[category] = max(sigma, MIN_STAT_STANDARD_DEVIATION)

    return category_sigmas


# ============================================================================
# CONVEXITY DIAGNOSTICS
# ============================================================================


def compute_category_regime(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    gradient: dict[str, float],
) -> pd.DataFrame:
    """Diagnose which categories are in the convex (undervalued) regime.

    The EW surface has curvature: when deeply losing a category (z < -0.5),
    the linear gradient underestimates the true marginal value of improvement.
    This is because Φ(z) is convex for z < 0 — the steepest gains come from
    climbing out of a deep hole.

    For each category, computes a "convexity ratio" — how much the actual EW
    gain from a typical perturbation exceeds the linear prediction. Ratio > 1
    means the gradient undervalues improvement; ratio < 1 means it overvalues.

    Returns:
        DataFrame with columns: category, avg_z, gradient, hessian_diag,
        convexity_ratio, regime. Sorted by convexity_ratio descending.
    """
    rows = []
    for cat in ALL_CATEGORIES:
        sigma = max(category_sigmas[cat], MIN_STAT_STANDARD_DEVIATION)
        denom = sigma * np.sqrt(2)

        zs = []
        for opp_totals in opponent_totals.values():
            if cat in NEGATIVE_CATEGORIES:
                z = (opp_totals[cat] - my_totals[cat]) / denom
            else:
                z = (my_totals[cat] - opp_totals[cat]) / denom
            zs.append(z)

        avg_z = float(np.mean(zs))
        hessian_diag = float(sum(-z * stats.norm.pdf(z) / denom**2 for z in zs))

        # Convexity ratio: ratio of actual ΔEW / linear ΔEW for a 0.5σ perturbation
        # (representative of a single good roster move in that category)
        delta_stat = 0.5 * sigma
        linear_dew = gradient[cat] * delta_stat
        if cat in NEGATIVE_CATEGORIES:
            delta_stat_for_ew = -delta_stat
        else:
            delta_stat_for_ew = delta_stat

        actual_dew = 0.0
        for z in zs:
            new_z = z + delta_stat_for_ew / denom
            actual_dew += stats.norm.cdf(new_z) - stats.norm.cdf(z)

        ratio = float(actual_dew / linear_dew) if abs(linear_dew) > 1e-10 else 1.0

        if avg_z < -0.5:
            regime = "convex (undervalued)"
        elif avg_z > 0.5:
            regime = "concave (overvalued)"
        else:
            regime = "near-linear"

        rows.append(
            {
                "category": cat,
                "avg_z": avg_z,
                "gradient": gradient[cat],
                "hessian_diag": hessian_diag,
                "convexity_ratio": ratio,
                "regime": regime,
            }
        )

    df = pd.DataFrame(rows).sort_values("convexity_ratio", ascending=False)
    return df.reset_index(drop=True)


# ============================================================================
# MONTE CARLO STANDINGS SIMULATION
# ============================================================================


def simulate_standings(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    n_sims: int = 20_000,
    seed: int = 0,
) -> dict:
    """Simulate final roto standings; turn EW into win/placement probabilities.

    EW (= expected standing points − 10) is risk-neutral: it says nothing about
    P(win the league), which is what actually matters — especially late in the
    season, when a trailing team should prefer variance and a leading team
    should avoid it. This simulator samples each team's final category totals,
    scores roto points, and reports the full placement distribution.

    Model: team c-totals ~ Normal(total_c, σ_c), independent across teams —
    EXACTLY the distributional assumptions of the analytical model (a pairwise
    gap then has SD σ_c·√2). Consequently E[my standing points] from the
    simulation converges to 10 + EW; this identity is the consistency test.

    Args:
        my_totals: My (season) category totals.
        opponent_totals: {opp_id: totals} for each opponent.
        category_sigmas: σ_c per category (per-team projection uncertainty).
        n_sims: Number of simulated seasons.
        seed: RNG seed (fixed for reproducible dashboards).

    Returns:
        {
            'expected_points': {team_idx: float},   # 0 = me, then opp ids order
            'p_win': {team_idx: float},             # P(strictly most points; ties split)
            'p_top2': {team_idx: float},
            'my_points_quantiles': {q: float},      # 5/25/50/75/95th percentiles
            'team_labels': list[str],               # 'me', 'opp_1', ...
        }
    """
    opp_ids = sorted(opponent_totals.keys())
    team_labels = ["me"] + [f"opp_{i}" for i in opp_ids]
    n_teams = len(team_labels)

    rng = np.random.default_rng(seed)
    total_points = np.zeros((n_sims, n_teams))

    for cat in ALL_CATEGORIES:
        sigma = max(category_sigmas[cat], MIN_STAT_STANDARD_DEVIATION)
        means = np.array([my_totals[cat]] + [opponent_totals[i][cat] for i in opp_ids])
        draws = rng.normal(loc=means, scale=sigma, size=(n_sims, n_teams))
        if cat in NEGATIVE_CATEGORIES:
            draws = -draws
        # Standing points: 1 + number of teams strictly worse. Continuous
        # draws make ties probability-zero, so a double-argsort rank is exact.
        ranks = draws.argsort(axis=1).argsort(axis=1)  # 0 = worst
        total_points += ranks + 1

    row_max = total_points.max(axis=1, keepdims=True)
    is_best = total_points == row_max
    # Split ties in final points evenly among the tied teams.
    win_credit = is_best / is_best.sum(axis=1, keepdims=True)

    # Top-2: points >= second-highest of the row.
    sorted_pts = np.sort(total_points, axis=1)
    second_best = sorted_pts[:, -2:-1]
    p_top2 = (total_points >= second_best).mean(axis=0)

    expected_points = total_points.mean(axis=0)
    p_win = win_credit.mean(axis=0)
    my_q = {q: float(np.percentile(total_points[:, 0], q)) for q in (5, 25, 50, 75, 95)}

    return {
        "expected_points": {i: float(v) for i, v in enumerate(expected_points)},
        "p_win": {i: float(v) for i, v in enumerate(p_win)},
        "p_top2": {i: float(v) for i, v in enumerate(p_top2)},
        "my_points_quantiles": my_q,
        "team_labels": team_labels,
    }


# ============================================================================
# STANDINGS
# ============================================================================


def compute_standings(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """Projected roto standings: rank and standing points per category.

    Returns:
        DataFrame with columns:
            category, my_value, opp_1, ..., opp_6, my_rank, wins

        my_rank: 1 = first place, 7 = last place.
        wins: number of opponents I beat (0–6).
        For negative categories (ERA, WHIP), lower value = better rank.
    """
    rows = []

    for cat in ALL_CATEGORIES:
        row: dict = {"category": cat, "my_value": my_totals[cat]}

        all_values = [my_totals[cat]]
        for opp_id in sorted(opponent_totals.keys()):
            row[f"opp_{opp_id}"] = opponent_totals[opp_id][cat]
            all_values.append(opponent_totals[opp_id][cat])

        if cat in NEGATIVE_CATEGORIES:
            sorted_vals = sorted(all_values)
            row["my_rank"] = sorted_vals.index(my_totals[cat]) + 1
            row["wins"] = sum(1 for v in all_values[1:] if my_totals[cat] < v)
        else:
            sorted_vals = sorted(all_values, reverse=True)
            row["my_rank"] = sorted_vals.index(my_totals[cat]) + 1
            row["wins"] = sum(1 for v in all_values[1:] if my_totals[cat] > v)

        rows.append(row)

    return pd.DataFrame(rows)
