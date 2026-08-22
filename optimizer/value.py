"""
The dynasty objective: V(p; β), one knob, denominated in Δ P(win the league).

    V(p; β) = Σ  β^t · u_t(p)          summed while β^t·|u_t| > EPSILON
              t≥0

    u_t(p) = Σ  prob_k · m_t(branch k)
             k

`m_t` is a player's marginal Δ P(win) in season t, scored against that season's
championship gradient (see `championship`). β is the ONLY free parameter:
β→0 is compete-now, β→1 is a fully patient rebuild.

There is NO horizon parameter. `survival` decays geometrically and `decay`
returns 0 past the fitted age band, so the sum converges even at β = 1.

WHAT CARRIES THE "MID IS WORST" DOCTRINE
----------------------------------------
Not a variance penalty. Three separable mechanisms, and only the third is a
preference:

  occupancy   A branch that never resolves holds the slot at m_t ≈ 0 for its
              whole horizon. `V_gross` does not charge rent for that; §6's
              differencing against the best available alternative does. So the
              mid outcome is expensive because of what it DISPLACES, and
              `net_value` is where that lands.
  option      `E[max(C,0)] ≥ max(E[C],0)`: the gap grows in the dispersion of
              the posterior AT THE DECISION TIME, not in the dispersion of the
              terminal outcome. A player who stays uncertain forever has
              variance and no option value.
  convexity   P is convex below the title cutoff, so a trailing team is paid
              for outcome variance. This one is automatic: `u_t` built from
              `exact_delta_p` per branch prices it to all orders, with no
              curvature coefficient to calibrate.

That last point is why this module has no Λ term. Evaluating P at each branch is
strictly better than a second-order expansion around the mean, and costs one
simulation per branch.
"""

import numpy as np
import pandas as pd

from .championship import (
    COUNTING_CATEGORIES,
    RATIO_VOLUME,
    exact_delta_p,
    score_line,
    vacate_slot,
)

# Stop summing once a season's discounted contribution falls below this. A
# numerical tolerance, not a modelling choice — the horizon is set by the data's
# own decay, not by a parameter.
EPSILON: float = 1e-6

# Loop backstop. `decay` returns 0 past its fitted band so the series
# terminates on its own; this only bounds a pathological input.
MAX_SEASONS: int = 30

# Scoring stats a projected line must carry for `score_line` and
# `totals_after_adding` to be well defined.
LINE_STATS: tuple[str, ...] = (
    "PA", "IP", "R", "HR", "RBI", "SB", "OPS", "W", "SV", "K", "ERA", "WHIP",
)

# Categories whose value scales with playing time, so survival multiplies them.
# Ratio categories are deliberately absent: their contribution is already scaled
# by the volume term (PA·(OPS − team_OPS)/team_PA), so multiplying the rate by
# survival as well would apply the haircut twice. A player who does not play has
# PA = 0, which zeroes his ratio contribution on its own — he does not acquire a
# worse OPS.
VOLUME_SCALED: tuple[str, ...] = ("PA", "IP", *COUNTING_CATEGORIES)


def project_line(
    line: dict[str, float],
    from_age: int,
    years: int,
    role: str,
    decay_lookup,
    survival_lookup,
) -> dict[str, float]:
    """Age one line forward `years` seasons and weight it by survival.

    Counting stats take BOTH decays — their own rate curve and the volume
    curve — because a count is a rate times playing time. Applying only the rate
    curve is how a model overvalues speed: stolen-base rate is nearly flat from
    22 to 26 while plate-appearance attrition is steepest for exactly those
    profiles.

    Args:
        line: Season-0 line, keys from LINE_STATS.
        from_age: Age in season 0. Must be inside the decay table's band.
        years: Seasons forward. 0 returns the line unchanged.
        role: "hitter", "starter" or "reliever" — the decay/survival key.
        decay_lookup: (age, years, category, role) -> cumulative factor.
        survival_lookup: (age, years, role) -> probability.

    Returns:
        A new line dict, same keys.
    """
    assert years >= 0, f"project_line: years must be >= 0, got {years}."
    if years == 0:
        return dict(line)

    volume_factor = decay_lookup(from_age, years, "VOL", role)
    survival = survival_lookup(from_age, years, role)
    # Past the fitted age band decay returns 0.0, which correctly zeroes the
    # whole line rather than extrapolating a curve that was never estimated.
    if volume_factor <= 0.0 or survival <= 0.0:
        return {key: 0.0 for key in line}

    out: dict[str, float] = {}
    for key, value in line.items():
        # Zero stays zero, and skipping the lookup is not just an optimisation:
        # the silver table fills the opposite player type's categories with 0.0,
        # so a hitter's line carries W/SV/K and a pitcher's carries PA/OPS. There
        # is no hitter curve for W, and asking for one is a crash.
        if value == 0.0:
            out[key] = 0.0
        elif key in ("PA", "IP"):
            out[key] = value * volume_factor * survival
        elif key in COUNTING_CATEGORIES:
            rate_factor = decay_lookup(from_age, years, key, role)
            out[key] = value * rate_factor * volume_factor * survival
        elif key in RATIO_VOLUME:
            out[key] = value * decay_lookup(from_age, years, key, role)
        else:
            out[key] = value
    return out


def annualize(line: dict[str, float], season_fraction_remaining: float) -> dict:
    """Scale a rest-of-season line up to a full season.

    Season 0 is the REMAINDER of the current season; every later season is a
    whole one. The projection feeds are rest-of-season, so using them unscaled
    for season 3 values a full year at a fifth of its production. In August that
    is a 5x error on every future season, and it silently favours whoever has
    the most games left rather than whoever is best.

    Only volume-bearing categories scale. Rates do not: a player's OPS over the
    remaining month is his OPS over a full season, not a fifth of it.
    """
    assert 0.0 < season_fraction_remaining <= 1.0, (
        f"annualize: season_fraction_remaining must lie in (0, 1], got "
        f"{season_fraction_remaining}. It comes from "
        f"optimizer.config.season_fraction_remaining()."
    )
    scale = 1.0 / season_fraction_remaining
    return {
        key: (value * scale if key in VOLUME_SCALED else value)
        for key, value in line.items()
    }


def single_branch(
    ros_line: dict[str, float],
    age: int,
    role: str,
    season_fraction_remaining: float,
) -> tuple[list[dict], list[dict]]:
    """Branches for a player already in the majors: one certain outcome.

    Returns (season_0_branches, later_branches). They differ ONLY in that
    season 0 uses the rest-of-season line and later seasons use the annualized
    one — see `annualize`. Keeping them separate is deliberate; a single branch
    list cannot carry two horizons.
    """
    now = [{"prob": 1.0, "line": dict(ros_line), "arrive": 0, "arrive_age": age,
            "role": role}]
    later = [{"prob": 1.0, "line": annualize(ros_line, season_fraction_remaining),
              "arrive": 0, "arrive_age": age, "role": role}]
    return now, later


def spliced_payoffs(
    now_branches: list[dict],
    later_branches: list[dict],
    context: dict,
    decay_lookup,
    survival_lookup,
    n_seasons: int,
) -> np.ndarray:
    """u_t with season 0 on the rest-of-season horizon and t>=1 annualized.

    Args:
        now_branches, later_branches: As returned by `single_branch`, or built
            the same way from an outcome mixture.
        context: From `championship.build_season_context`.

    Returns:
        Length-n_seasons payoff array.
    """
    head = branch_payoffs(
        now_branches,
        context["gradients"][:1],
        context["references"][:1],
        decay_lookup,
        survival_lookup,
        1,
    )
    tail = branch_payoffs(
        later_branches,
        context["gradients"],
        context["references"],
        decay_lookup,
        survival_lookup,
        n_seasons,
    )
    return np.concatenate([head, tail[1:]])


def branch_payoffs(
    branches: list[dict],
    gradients: list[dict[str, float]],
    reference_totals: list[dict[str, float]],
    decay_lookup,
    survival_lookup,
    n_seasons: int,
) -> np.ndarray:
    """Per-season linear payoff u_t, expectation taken over outcome branches.

    Args:
        branches: One dict per outcome branch:
            {'prob': float, 'line': dict, 'arrive': int, 'arrive_age': int,
             'role': str}
            `arrive` is the season index the line starts paying (0 for a player
            already in the majors). `arrive_age` is his age THEN — a prospect is
            aged from his projected arrival age, never from his current age,
            because the decay curves do not exist below age 20.
        gradients: Championship gradient per season, index t.
        reference_totals: Reference team per season, index t.
        n_seasons: Length of both lists above.

    Returns:
        Array of length n_seasons. Probability units.
    """
    assert len(gradients) == n_seasons and len(reference_totals) == n_seasons, (
        f"branch_payoffs: need one gradient and one reference per season; got "
        f"{len(gradients)} gradients, {len(reference_totals)} references for "
        f"{n_seasons} seasons."
    )
    total_probability = sum(float(b["prob"]) for b in branches)
    assert abs(total_probability - 1.0) < 1e-6, (
        f"branch_payoffs: branch probabilities sum to {total_probability:.6f}, "
        f"not 1.0. The never-arrives branch must be present explicitly (an "
        f"all-zero line); missing mass silently discounts the whole player."
    )

    payoffs = np.zeros(n_seasons, dtype=float)
    for branch in branches:
        probability = float(branch["prob"])
        if probability <= 0.0:
            continue
        arrive = int(branch["arrive"])
        arrive_age = int(branch["arrive_age"])
        role = branch["role"]
        for season in range(arrive, n_seasons):
            projected = project_line(
                branch["line"],
                arrive_age,
                season - arrive,
                role,
                decay_lookup,
                survival_lookup,
            )
            payoffs[season] += probability * score_line(
                projected, gradients[season], reference_totals[season]
            )
    return payoffs


def exact_branch_payoffs(
    branches: list[dict],
    opponent_totals: dict[int, dict[str, float]],
    sigmas_by_season: list[dict[str, float]],
    vacant_totals_by_season: list[dict[str, float]],
    decay_lookup,
    survival_lookup,
    n_seasons: int,
    **simulation_kwargs,
) -> np.ndarray:
    """Per-season payoff u_t evaluated EXACTLY, one simulation per branch.

    Same contract as `branch_payoffs`, but calls `exact_delta_p` instead of the
    linear `score_line`. Prices the win curve's convexity to all orders, which
    is what makes a high-variance prospect worth more than his mean to a
    trailing team — no curvature coefficient required.

    Costs `n_branches × n_seasons` simulations, so this is the candidate path.
    Screen with `branch_payoffs` first.
    """
    payoffs = np.zeros(n_seasons, dtype=float)
    baselines: list[float | None] = [None] * n_seasons
    for branch in branches:
        probability = float(branch["prob"])
        if probability <= 0.0:
            continue
        arrive = int(branch["arrive"])
        arrive_age = int(branch["arrive_age"])
        role = branch["role"]
        for season in range(arrive, n_seasons):
            projected = project_line(
                branch["line"],
                arrive_age,
                season - arrive,
                role,
                decay_lookup,
                survival_lookup,
            )
            delta = exact_delta_p(
                vacant_totals_by_season[season],
                opponent_totals,
                sigmas_by_season[season],
                projected,
                baseline=baselines[season],
                **simulation_kwargs,
            )
            payoffs[season] += probability * delta
    return payoffs


def player_value(payoffs: np.ndarray, beta: float) -> float:
    """V = Σ β^t · u_t, truncated where the discounted term falls under EPSILON.

    β = 1.0 is legal and finite: survival decays geometrically, so the series
    converges without a horizon parameter. The data supplies its own discount.
    """
    assert 0.0 < beta <= 1.0, (
        f"player_value: beta must lie in (0, 1], got {beta}. 0 would value only "
        f"the current season and is better expressed as beta just above 0; "
        f"above 1 would value the future more than the present."
    )
    # Sum every season. Do NOT stop early on a small term: a prospect's
    # pre-arrival seasons are exactly 0.0, so breaking at the first sub-epsilon
    # discounted term exits at t=1 and discards his entire career. That bug
    # zeroed every prospect -- precisely the players this model exists to value.
    # MAX_SEASONS already bounds the work, and `survival` drives the tail to
    # zero on its own, which is what makes beta = 1 converge.
    horizon = min(len(payoffs), MAX_SEASONS)
    weights = np.power(beta, np.arange(horizon, dtype=float))
    return float(np.dot(weights, np.asarray(payoffs[:horizon], dtype=float)))


def net_value(
    payoffs: np.ndarray, alternative_payoffs: np.ndarray, beta: float
) -> float:
    """V_net = V_gross(p) − V_gross(best available alternative for the slot).

    Opportunity cost is differenced HERE, at the level of V, and never as a
    per-season rent inside u_t. A per-season rent is not dimensionally
    homogeneous across pools: a minor-league slot generates zero current-season
    probability because it cannot be started, so its cost is an option on future
    seasons while a major-league slot's is a per-season flow. Differencing V
    sidesteps that entirely.

    This is where "a prospect who becomes a mid major leaguer is the worst
    asset" becomes arithmetic rather than doctrine: his own V_gross is small
    because m_t ≈ 0 across a long occupancy, so V_net goes NEGATIVE exactly when
    the best available alternative has positive V_gross. That is a testable
    condition, not an assertion.
    """
    return player_value(payoffs, beta) - player_value(alternative_payoffs, beta)


def dominates(payoffs_a: np.ndarray, payoffs_b: np.ndarray) -> bool:
    """True when A is at least as good as B in EVERY season, and better in one.

    Parameter-free: if it holds, A beats B at every β, so no calibration and no
    argument about impatience can reverse it. Run this before tuning anything —
    it prunes without a single judgement call.
    """
    length = max(len(payoffs_a), len(payoffs_b))
    a = np.pad(payoffs_a, (0, length - len(payoffs_a)))
    b = np.pad(payoffs_b, (0, length - len(payoffs_b)))
    return bool(np.all(a >= b) and np.any(a > b))


def breakeven_beta(
    payoffs_a: np.ndarray, payoffs_b: np.ndarray, tolerance: float = 1e-9
) -> dict:
    """Where in β does the ranking of A and B flip?

    V_A(β) − V_B(β) = Σ β^t·(u_t(A) − u_t(B)) is a polynomial in β, so by
    Descartes' rule the number of positive roots is bounded by the number of
    SIGN CHANGES in that coefficient sequence. One sign change means exactly one
    crossing, and a single break-even β is then a complete and honest summary.

    Prospect-versus-veteran has coefficient pattern (−, −, +, +, +): the veteran
    leads early, the prospect leads late. One sign change, so the report is
    valid precisely in the case we care about. Two players can otherwise cross
    more than once, and a lone number would then be a lie.

    Returns:
        {'sign_changes': int,
         'roots': list[float]      roots inside (0, 1], ascending
         'unique': bool            True iff exactly one sign change
         'summary': str}
    """
    length = max(len(payoffs_a), len(payoffs_b))
    a = np.pad(payoffs_a, (0, length - len(payoffs_a)))
    b = np.pad(payoffs_b, (0, length - len(payoffs_b)))
    difference = a - b

    significant = difference[np.abs(difference) > tolerance]
    signs = np.sign(significant)
    sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

    # numpy.polynomial takes coefficients in ascending degree, which is exactly
    # the season ordering, so no reversal.
    if np.all(np.abs(difference) <= tolerance):
        return {
            "sign_changes": 0,
            "roots": [],
            "unique": False,
            "summary": "identical payoffs at every horizon; no crossing",
        }
    roots = np.polynomial.polynomial.polyroots(difference)
    real_roots = sorted(
        float(r.real)
        for r in np.atleast_1d(roots)
        if abs(r.imag) < 1e-9 and 0.0 < r.real <= 1.0
    )

    unique = sign_changes == 1
    if not real_roots:
        leader = "A" if player_value(a, 1.0) > player_value(b, 1.0) else "B"
        summary = f"no crossing in (0, 1]; {leader} wins at every beta"
    elif unique:
        summary = f"B passes A at beta = {real_roots[0]:.3f}"
    else:
        summary = (
            f"{sign_changes} sign changes and {len(real_roots)} crossings "
            f"at {[round(r, 3) for r in real_roots]}; a single break-even "
            f"would misrepresent this pair"
        )
    return {
        "sign_changes": sign_changes,
        "roots": real_roots,
        "unique": unique,
        "summary": summary,
    }


def pareto_frontier(payoff_matrix: dict[str, np.ndarray]) -> list[str]:
    """Names not dominated by any other, i.e. optimal at SOME β.

    Reachability caveat, so nobody over-reads this: weighted-sum scalarization
    with weights on the moment curve (1, β, β², …) exposes only the upper-right
    CONVEX hull. A player sitting in a non-convex dent of the Pareto frontier is
    top-ranked at no single β. That invalidates only claims of the form "p is
    never the single best player" — never "p is not worth acquiring", because
    roster construction picks 28+10 additive slots and a portfolio optimum
    mixes, which fills the dents back in.
    """
    names = list(payoff_matrix)
    keep = []
    for name in names:
        if not any(
            other != name and dominates(payoff_matrix[other], payoff_matrix[name])
            for other in names
        ):
            keep.append(name)
    return keep


def beta_sweep(
    payoff_matrix: dict[str, np.ndarray], betas: tuple[float, ...]
) -> pd.DataFrame:
    """Rank every player at each β. The report, rather than one number.

    Returns:
        Wide frame indexed by name, one column per β holding V, plus `rank_at`
        columns. A player whose rank is stable across the sweep is a decision
        you can make without settling β; one whose rank swings is a decision
        that depends on a posture you have to state.
    """
    frame = pd.DataFrame(
        {
            f"V@{beta:g}": {
                name: player_value(payoffs, beta)
                for name, payoffs in payoff_matrix.items()
            }
            for beta in betas
        }
    )
    for beta in betas:
        frame[f"rank@{beta:g}"] = frame[f"V@{beta:g}"].rank(ascending=False)
    rank_columns = [f"rank@{beta:g}" for beta in betas]
    frame["rank_swing"] = frame[rank_columns].max(axis=1) - frame[
        rank_columns
    ].min(axis=1)
    return frame.sort_values(f"V@{betas[len(betas) // 2]:g}", ascending=False)
