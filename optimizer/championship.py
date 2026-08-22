"""
Championship-probability scoring: the gradient G, and exact per-player deltas.

The dynasty objective is denominated in Δ P(win the league). This module is the
only place that quantity is produced; everything downstream consumes it.

WHY THERE IS NO SCALAR dP/dEW
-----------------------------
P(win) is a function of the ten-vector of category totals, not of the scalar EW.
Two rosters with identical EW and different category profiles have different
P(win) — punting saves is not interchangeable with spreading thin. So dP/dEW is
undefined until a perturbation direction is fixed, and a player IS a direction:
he adds HR and R and no SV. We therefore differentiate P against the totals
directly:

    G_c = dP(win) / d(my_c)                     `championship_gradient`

and score a player as Σ_c G_c · stat_c(p) + ratio terms. That expression is
exactly `player_scoring.add_mew`'s formula, so passing G in place of the EW
gradient produces MEW denominated in Δ P(win) with no new scoring code.

SCREEN WITH G, THEN EVALUATE EXACTLY
------------------------------------
G is a first-order object and P is distinctly non-linear near the cutoff — which
is the whole reason variance has value. Rather than bolt a Taylor curvature term
onto the linear score, evaluate P at each outcome branch directly:

    exact_delta_p(totals, line)   ->  P(totals + line) - P(totals)

An expectation of `exact_delta_p` over a player's outcome mixture captures the
convexity to all orders and needs no curvature coefficient at all. It costs one
simulation per branch, so it is for candidates; G screens the other seven
thousand. This mirrors the repo's existing screen-then-exact split (§4a, W13).

COMMON RANDOM NUMBERS ARE MANDATORY
-----------------------------------
Every simulation here runs on a FIXED seed. p_win is a mean of indicators, so an
independent-seed difference of two runs is dominated by Monte Carlo noise: at
n_sims=50_000 each estimate carries se ~ 0.001, while the differences we need to
resolve are of that same order. On a common seed the two runs share every draw
and differ only by the shift, so the difference is far better conditioned than
either estimate. `gradient_noise` measures what is left.
"""

import numpy as np
import pandas as pd

from .config import ALL_CATEGORIES, NEGATIVE_CATEGORIES
from .win_model import simulate_standings

# Categories that accumulate, versus rates that are volume-weighted averages.
COUNTING_CATEGORIES: tuple[str, ...] = ("R", "HR", "RBI", "SB", "W", "SV", "K")
# Ratio category -> the volume it is weighted by.
RATIO_VOLUME: dict[str, str] = {"OPS": "PA", "ERA": "IP", "WHIP": "IP"}

# Simulation draws per evaluation. Higher than win_model's 20_000 default: we
# are differencing two runs, and the quantities of interest are small.
DEFAULT_N_SIMS: int = 50_000
# One fixed seed for every call in a scoring pass. See the module docstring.
DEFAULT_SEED: int = 20260822

# Finite-difference step as a fraction of the category's own sigma. Large enough
# that enough simulated seasons flip to resolve the difference above the
# quantisation floor, small enough to stay local.
DEFAULT_STEP_FRAC: float = 0.25

# Below this win probability the gradient is numerically indistinguishable from
# noise, because essentially no simulated season is near the boundary. That is
# not a bug — a hopeless team's current season genuinely carries no leverage —
# but propagating noise as if it were signal is. See `championship_gradient`.
DEGENERATE_P_FLOOR: float = 0.002

# Objectives `simulate_standings` reports per team.
#
#   p_win           the sharp objective, and the one the model is built on.
#   p_top2          usable when p_win is decided but placement still is not.
#   expected_points EXPLICITLY RISK-NEUTRAL. Roto standing points are close to
#                   linear in team strength, so this objective has almost no
#                   curvature — which means the entire "a trailing team is paid
#                   for variance" mechanism vanishes under it. Use it to answer
#                   "what maximises my finish", never to value upside.
OBJECTIVES: frozenset[str] = frozenset({"p_win", "p_top2", "expected_points"})


def win_probability(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    objective: str = "p_win",
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> float:
    """My probability under `objective`, on a fixed seed.

    Args:
        my_totals: My category totals. May carry PA/IP; only the ten scoring
            categories are read.
        opponent_totals: {opp_id: totals}.
        category_sigmas: Per-category sigma.
        objective: "p_win" or "p_top2". p_win is the sharp objective; p_top2 is
            useful when p_win is degenerate (see DEGENERATE_P_FLOOR) and the
            league pays for placement.
        n_sims: Simulation draws.
        seed: Fixed across a scoring pass — never vary it between the two halves
            of a difference.

    Returns:
        Probability in [0, 1].
    """
    assert objective in OBJECTIVES, (
        f"win_probability: objective must be one of {sorted(OBJECTIVES)}, got "
        f"{objective!r}."
    )
    result = simulate_standings(
        my_totals, opponent_totals, category_sigmas, n_sims=n_sims, seed=seed
    )
    return float(result[objective][0])


def totals_after_adding(
    my_totals: dict[str, float], line: dict[str, float], sign: float = 1.0
) -> dict[str, float]:
    """My totals after adding (or removing) one player's season line.

    Counting categories add directly. Ratio categories are re-derived as
    volume-weighted averages, NOT approximated — this is the exact update, so it
    stays correct for a player whose volume is large relative to the team's.
    The first-order form in `add_mew` is only valid for small volumes, and a
    starting pitcher is 15-20% of a roto team's innings.

    Args:
        my_totals: Totals including 'PA' and 'IP'.
        line: One player's line. Missing categories are treated as 0.
        sign: +1 to add, -1 to remove.

    Returns:
        A new totals dict, same keys as `my_totals`.
    """
    for volume in ("PA", "IP"):
        assert volume in my_totals, (
            f"totals_after_adding: my_totals is missing {volume!r}. Ratio "
            f"categories are volume-weighted, so the exact update needs team "
            f"PA and IP. Use compute_totals_for_starters, which supplies both."
        )
    out = dict(my_totals)

    for category in COUNTING_CATEGORIES:
        out[category] = my_totals[category] + sign * float(line.get(category, 0.0))

    for category, volume_key in RATIO_VOLUME.items():
        team_volume = float(my_totals[volume_key])
        player_volume = sign * float(line.get(volume_key, 0.0))
        new_volume = team_volume + player_volume
        if new_volume <= 0.0:
            # Removing the only contributor leaves the rate undefined. Hold the
            # current value rather than inventing one; the caller is asking
            # about an empty roster half, which no sane comparison reaches.
            out[category] = my_totals[category]
            continue
        weighted = my_totals[category] * team_volume + player_volume * float(
            line.get(category, 0.0)
        )
        out[category] = weighted / new_volume

    out["PA"] = my_totals["PA"] + sign * float(line.get("PA", 0.0))
    out["IP"] = my_totals["IP"] + sign * float(line.get("IP", 0.0))
    return out


# Resolution floor. p_win is a mean of n_sims indicators, so a difference
# smaller than a handful of flipped seasons is not measurable. Differences below
# this are noise regardless of how many decimal places they print with.
def resolution_floor(n_sims: int = DEFAULT_N_SIMS) -> float:
    """Smallest Δ P that is distinguishable from zero at this many draws."""
    return 5.0 / float(n_sims)


def exact_delta_p(
    vacant_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    line: dict[str, float],
    baseline: float | None = None,
    objective: str = "p_win",
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Δ P(objective) from putting one player's line into a VACANT slot.

    Exact, not linearised: captures the win curve's curvature to all orders, so
    no separate variance coefficient is needed. An expectation of this over a
    player's outcome branches already prices convexity.

    `vacant_totals` MUST already have the slot's incumbent removed — use
    `vacate_slot`. Passing a full roster's totals measures "add a nineteenth
    starter to an eighteen-slot lineup", which is not a move anyone can make and
    inflates every player enormously: against real 2026 rest-of-season totals it
    reported a single shortstop lifting p_win from 5% to 36%.

    Args:
        vacant_totals: Totals with the target slot empty.
        line: The candidate's line, on the SAME horizon as the totals. Adding a
            full-season line to rest-of-season totals compares different
            quantities.
        baseline: P at `vacant_totals`, if already computed. Pass it — otherwise
            a scoring pass recomputes the same number once per player.

    Returns:
        P(slot filled by this player) − P(slot empty), in probability units.
        Compare against `resolution_floor(n_sims)` before believing a small
        difference.
    """
    if baseline is None:
        baseline = win_probability(
            vacant_totals, opponent_totals, category_sigmas, objective, n_sims, seed
        )
    after = win_probability(
        totals_after_adding(vacant_totals, line),
        opponent_totals,
        category_sigmas,
        objective,
        n_sims,
        seed,
    )
    return after - baseline


def vacate_slot(
    my_totals: dict[str, float], incumbent_line: dict[str, float]
) -> dict[str, float]:
    """My totals with the incumbent's contribution removed.

    The reference point for valuing anyone who could occupy that slot. Two
    candidates compared against the same vacated totals are directly
    comparable; compared against the full roster they are not.
    """
    return totals_after_adding(my_totals, incumbent_line, sign=-1.0)


def swap_delta_p(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    incumbent_line: dict[str, float],
    candidate_line: dict[str, float],
    objective: str = "p_win",
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Δ P from replacing the incumbent with the candidate in one slot.

    The only quantity a manager can actually act on. Two simulations, because
    P(vacated + incumbent) is by construction P(my_totals) and needs no
    recomputation.

    Returns:
        P(after swap) − P(now). Positive means make the move.
    """
    before = win_probability(
        my_totals, opponent_totals, category_sigmas, objective, n_sims, seed
    )
    vacated = vacate_slot(my_totals, incumbent_line)
    after = win_probability(
        totals_after_adding(vacated, candidate_line),
        opponent_totals,
        category_sigmas,
        objective,
        n_sims,
        seed,
    )
    return after - before


def score_line(
    line: dict[str, float],
    gradient: dict[str, float],
    reference_totals: dict[str, float],
) -> float:
    """Linear Δ P(win) for one player line. The SCREENING score.

    This is `add_mew`'s formula with a championship gradient in place of the EW
    gradient, so the result is in probability units:

        Σ_c G_c·stat_c  +  Σ_ratio G_r · vol_p · (rate_p − team_rate) / team_vol

    First-order, and P is convex below the cutoff, so this UNDERSTATES a good
    player. Measured against real 2026 rest-of-season totals at p_win = 5.3%,
    it came in 20-46% below `exact_delta_p`, and the shortfall grew with the
    size of the addition. Use it to rank and to prune, never to quote a number.

    Args:
        line: One player's line on the same horizon as `reference_totals`.
        gradient: From `championship_gradient`.
        reference_totals: The team the ratio terms are differenced against.
            Must carry 'PA' and 'IP'.
    """
    for volume in ("PA", "IP"):
        assert volume in reference_totals, (
            f"score_line: reference_totals is missing {volume!r}; the ratio "
            f"terms divide by it."
        )
    total = 0.0
    for category in COUNTING_CATEGORIES:
        total += gradient[category] * float(line.get(category, 0.0))
    for category, volume_key in RATIO_VOLUME.items():
        team_volume = float(reference_totals[volume_key])
        if team_volume <= 0.0:
            continue
        total += (
            gradient[category]
            * float(line.get(volume_key, 0.0))
            * (float(line.get(category, 0.0)) - float(reference_totals[category]))
            / team_volume
        )
    return total


def championship_gradient(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    objective: str = "p_win",
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    step_frac: float = DEFAULT_STEP_FRAC,
) -> tuple[dict[str, float], dict]:
    """G_c = dP/d(my_c) per category, by central finite difference.

    Costs 1 + 2·|C| = 21 simulations. Every one uses the same `seed`.

    Perturbs raw category units and lets the simulator's own
    NEGATIVE_CATEGORIES handling set each sign, so G_ERA and G_WHIP come out
    negative. Do NOT pre-negate.

    Returns:
        (gradient, diagnostics) where gradient maps each of the ten categories
        to dP/d(unit). diagnostics carries:
            'baseline'   P at my_totals
            'degenerate' True when baseline is below DEGENERATE_P_FLOOR, in
                         which case EVERY gradient entry is forced to exactly
                         0.0 rather than reporting noise. A team that cannot win
                         has no current-season leverage; that is the honest
                         answer, and it is the caller's cue to lean on future
                         seasons instead of this season's noise.
            'steps'      the δ_c actually used
            'raw'        the unforced gradient, so a degenerate case can still
                         be inspected
    """
    assert step_frac > 0.0, (
        f"championship_gradient: step_frac must be positive, got {step_frac}."
    )
    baseline = win_probability(
        my_totals, opponent_totals, category_sigmas, objective, n_sims, seed
    )

    gradient: dict[str, float] = {}
    steps: dict[str, float] = {}
    for category in ALL_CATEGORIES:
        step = step_frac * float(category_sigmas[category])
        assert step > 0.0, (
            f"championship_gradient: step for {category} is {step}; sigma is "
            f"{category_sigmas[category]}. A zero sigma makes the difference "
            f"undefined — check estimate_projection_uncertainty."
        )
        steps[category] = step

        up = dict(my_totals)
        up[category] = my_totals[category] + step
        down = dict(my_totals)
        down[category] = my_totals[category] - step

        p_up = win_probability(
            up, opponent_totals, category_sigmas, objective, n_sims, seed
        )
        p_down = win_probability(
            down, opponent_totals, category_sigmas, objective, n_sims, seed
        )
        gradient[category] = (p_up - p_down) / (2.0 * step)

    degenerate = baseline < DEGENERATE_P_FLOOR
    raw = dict(gradient)
    if degenerate:
        gradient = {category: 0.0 for category in ALL_CATEGORIES}
        print(
            f"championship_gradient: baseline {objective}={baseline:.5f} is "
            f"below the {DEGENERATE_P_FLOOR} floor; forcing G to zero. This "
            f"season carries no measurable leverage — score on future seasons."
        )
    else:
        wrong_sign = [
            category
            for category in ALL_CATEGORIES
            if category in NEGATIVE_CATEGORIES
            and gradient[category] > 0.0
            or category not in NEGATIVE_CATEGORIES
            and gradient[category] < 0.0
        ]
        if wrong_sign:
            print(
                f"championship_gradient: WARNING sign violation in "
                f"{wrong_sign}. Either step_frac is below the simulation's "
                f"resolution or those races are fully decided. Raise n_sims or "
                f"step_frac and re-check before trusting these entries."
            )

    diagnostics = {
        "baseline": baseline,
        "objective": objective,
        "degenerate": degenerate,
        "steps": steps,
        "raw": raw,
        "n_sims": n_sims,
        "seed": seed,
    }
    return gradient, diagnostics


def gradient_noise(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    **kwargs,
) -> pd.DataFrame:
    """Re-estimate G across independent seeds and report the spread.

    A gradient whose seed-to-seed standard deviation is a large fraction of its
    own magnitude is not measuring anything. Run this once when the league state
    changes rather than trusting DEFAULT_N_SIMS blindly.

    Returns:
        One row per category: category, mean, sd, cv (sd/|mean|), n_seeds.
        Sort by cv descending; anything above ~0.2 needs more draws.
    """
    estimates = []
    for seed in seeds:
        gradient, _ = championship_gradient(
            my_totals, opponent_totals, category_sigmas, seed=seed, **kwargs
        )
        estimates.append(gradient)

    rows = []
    for category in ALL_CATEGORIES:
        values = np.array([e[category] for e in estimates], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "category": category,
                "mean": mean,
                "sd": sd,
                "cv": sd / abs(mean) if abs(mean) > 0 else np.nan,
                "n_seeds": len(values),
            }
        )
    return pd.DataFrame(rows).sort_values("cv", ascending=False).reset_index(drop=True)


def league_mean_totals(
    my_totals: dict[str, float], opponent_totals: dict[int, dict[str, float]]
) -> dict[str, float]:
    """The average team in this league, category by category.

    This is the nominal team the model scores future seasons against, because we
    do not know what roster we will own then. Ratio categories are averaged
    across teams rather than volume-pooled: each team is one competitor in the
    standings, so each counts once regardless of how many plate appearances it
    accumulated.
    """
    teams = [my_totals] + list(opponent_totals.values())
    keys = set(ALL_CATEGORIES) | {"PA", "IP"}
    return {
        key: float(np.mean([float(team[key]) for team in teams]))
        for key in keys
        if all(key in team for team in teams)
    }


def nominal_league(
    mean_totals: dict[str, float], n_opponents: int
) -> dict[int, dict[str, float]]:
    """Every opponent placed at the league mean. The future field.

    Future seasons must nominalise the OPPONENTS too, not just my own roster.
    Holding the field at its current totals while moving myself to the mean asks
    "what if I were average and everyone else stayed exactly as they are", which
    is not a coherent league: measured against real 2026 totals it put a
    league-mean team's p_win at 2.2% instead of the 1/7 = 14.3% that symmetry
    requires, because the frozen field still contained a 79.5% juggernaut. Every
    future season inherited that distortion and looked far worse than it should.

    With a symmetric field, p_win is 1/(n_opponents + 1) by construction, which
    is the peak of the leverage curve — precisely the "bubble team" that psi = 0
    is supposed to mean. Sigma inflation then flattens the gradient with horizon
    without moving the baseline, so the derived discount is clean.
    """
    assert n_opponents >= 1, (
        f"nominal_league: need at least one opponent, got {n_opponents}."
    )
    return {index: dict(mean_totals) for index in range(1, n_opponents + 1)}


def nominal_totals(
    my_totals: dict[str, float],
    mean_totals: dict[str, float],
    psi: float,
    horizon: int,
) -> dict[str, float]:
    """The team I am assumed to field `horizon` seasons out.

    Interpolates from my current roster toward the league mean:

        totals_t = mean + psi^t · (mine − mean)

    psi = 0.0 (the default, "A1") snaps straight to a league-mean team in every
    future season. That is NOT the neutral choice: a league-mean team sits at the
    peak of the leverage curve, so A1 makes every future season maximally
    valuable and tilts toward rebuilding before the impatience knob does
    anything. psi = 1.0 is persistence — you stay exactly who you are, which
    makes a bad team's future worthless too. Values in between are mean
    reversion, which is the defensible middle.

    A1's other consequence: at the league mean you sit at the win curve's
    inflection point, where variance has no first-order value. Future-season
    upside is then priced only through the option to abandon, not through
    convexity. Raising psi restores the convexity channel.
    """
    assert 0.0 <= psi <= 1.0, (
        f"nominal_totals: psi must lie in [0, 1], got {psi}. 0 is a league-mean "
        f"team every future season, 1 is my current roster forever."
    )
    assert horizon >= 0, f"nominal_totals: horizon must be >= 0, got {horizon}."
    weight = psi**horizon
    return {
        key: float(mean_totals[key] + weight * (my_totals[key] - mean_totals[key]))
        for key in mean_totals
        if key in my_totals
    }


def build_season_context(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    n_seasons: int,
    psi: float = 0.0,
    drift_variance: float = 0.35,
    objective: str = "p_win",
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """One gradient, reference team, field and sigma set per future season.

    Season 0 is MY actual league: my totals, the real opponents, this season's
    sigmas. Every later season is nominalised — me interpolated toward the mean
    by `psi`, the whole field AT the mean, sigmas widened by horizon.

    That asymmetry is the model's central claim. Season 0 answers "how much does
    production help the team I actually have, in the race I am actually in";
    later seasons answer "how much does production help a typical team in a
    typical race", because I do not know what roster I will own then.

    Returns:
        {'gradients', 'references', 'fields', 'sigmas', 'baselines',
         'degenerate', 'mean_totals'} — each list indexed by season.
        'degenerate' is a per-season bool list; a True entry means that season's
        gradient was forced to zero because the race is decided. For a team
        eliminated in the current season, entry 0 is True and the whole
        compete-now term is correctly zero, which makes β irrelevant for it.
    """
    mean_totals = league_mean_totals(my_totals, opponent_totals)
    nominal_field = nominal_league(mean_totals, len(opponent_totals))

    gradients, references, fields, sigmas = [], [], [], []
    baselines, degenerate = [], []
    for season in range(n_seasons):
        if season == 0:
            reference, field, sigma = my_totals, opponent_totals, category_sigmas
        else:
            reference = nominal_totals(my_totals, mean_totals, psi, season)
            field = nominal_field
            sigma = inflate_sigmas(category_sigmas, season, drift_variance)
        gradient, diagnostics = championship_gradient(
            reference, field, sigma, objective=objective, n_sims=n_sims, seed=seed
        )
        gradients.append(gradient)
        references.append(reference)
        fields.append(field)
        sigmas.append(sigma)
        baselines.append(diagnostics["baseline"])
        degenerate.append(diagnostics["degenerate"])

    print(
        f"season context: {n_seasons} seasons, objective={objective}, "
        f"psi={psi}, drift={drift_variance}; baselines "
        f"{[round(b, 4) for b in baselines]}"
    )
    if degenerate[0]:
        print(
            "  season 0 is DECIDED: its gradient is zero, so current-season "
            "production carries no championship value and beta cannot change "
            "that. Every ranking below is a statement about future seasons."
        )
    return {
        "gradients": gradients,
        "references": references,
        "fields": fields,
        "sigmas": sigmas,
        "baselines": baselines,
        "degenerate": degenerate,
        "mean_totals": mean_totals,
    }


def inflate_sigmas(
    category_sigmas: dict[str, float], horizon: int, drift_variance: float
) -> dict[str, float]:
    """Widen every sigma for a forecast `horizon` seasons out.

        sigma_t = sigma_0 · sqrt(1 + t · drift_variance)

    This is what makes distant seasons matter less BEFORE any impatience is
    applied: a wider sigma flattens Φ, so every entry of G shrinks with t. That
    is a derived discount, and it is not β.

    `drift_variance` absorbs two things it cannot separate — genuine forecast
    error, and the fact that a future roster is chosen rather than drawn. Treat
    it as a tuning knob, not an estimate.
    """
    assert horizon >= 0, f"inflate_sigmas: horizon must be >= 0, got {horizon}."
    assert drift_variance >= 0.0, (
        f"inflate_sigmas: drift_variance must be >= 0, got {drift_variance}. A "
        f"negative value would make distant seasons more certain than this one."
    )
    scale = float(np.sqrt(1.0 + horizon * drift_variance))
    return {category: sigma * scale for category, sigma in category_sigmas.items()}
