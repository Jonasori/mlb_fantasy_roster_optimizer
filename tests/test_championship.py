"""
Offline tests for championship-probability scoring.
No network. Per AGENTS.md: no classes, no fixtures, no mocking.

Several of these exist because the bug they guard actually shipped into a run:

  test_vacate_then_readd_is_identity      the slot-vacated baseline. Scoring
      against a full roster measured a 19-starter lineup against an 18-starter
      one and reported one shortstop lifting p_win from 5% to 36%.
  test_nominal_league_is_symmetric        freezing the opponents while
      nominalising myself put a league-mean team's p_win at 2.2% instead of 1/7,
      because the frozen field still contained a 79.5% juggernaut.
  test_ratio_update_is_exact_not_linear   a starting pitcher is 15-20% of a roto
      team's innings, so the first-order ratio form in add_mew is not accurate
      enough to value one.
"""

import numpy as np
import pytest

from optimizer.championship import (
    DEGENERATE_P_FLOOR,
    build_season_context,
    championship_gradient,
    exact_delta_p,
    inflate_sigmas,
    league_mean_totals,
    nominal_league,
    nominal_totals,
    resolution_floor,
    score_line,
    swap_delta_p,
    totals_after_adding,
    vacate_slot,
    win_probability,
)
from optimizer.config import ALL_CATEGORIES, NEGATIVE_CATEGORIES

# Small but not tiny: enough draws that a sign is stable, few enough to be fast.
_SIMS = 4000


def _totals(**overrides) -> dict[str, float]:
    base = {
        "R": 800.0, "HR": 220.0, "RBI": 780.0, "SB": 120.0, "OPS": 0.740,
        "W": 70.0, "SV": 60.0, "K": 1300.0, "ERA": 3.900, "WHIP": 1.250,
        "PA": 6500.0, "IP": 1300.0,
    }
    base.update(overrides)
    return base


def _field(n: int = 6, **overrides) -> dict[int, dict[str, float]]:
    return {i: _totals(**overrides) for i in range(1, n + 1)}


def _graded_field(n: int = 6) -> dict[int, dict[str, float]]:
    """A field spread across strengths, so mid-pack is a real place to be.

    An IDENTICAL field is degenerate for anything but the symmetry test: every
    opponent has the same totals, so a team even slightly behind loses every
    category to all six at once and p_win collapses to 0. Tests about being
    "somewhat behind" need a graded field or they land in the degenerate branch
    and measure nothing.

    Scaling every category together is equally useless: it manufactures one team
    strong at EVERYTHING, which drives everyone else's p_win to zero too. Real
    roto teams are strong in some categories and weak in others, so the strength
    offsets here are ROTATED across categories per team.
    """
    base = _totals()
    order = list(ALL_CATEGORIES)
    field = {}
    for index in range(1, n + 1):
        totals = dict(base)
        for position, category in enumerate(order):
            # Each team's advantage lands on a different set of categories.
            offset = 0.12 * np.cos(2 * np.pi * ((position + index) / len(order)))
            direction = -1.0 if category in NEGATIVE_CATEGORIES else 1.0
            totals[category] = base[category] * (1.0 + direction * offset)
        field[index] = totals
    return field


def _hitter_line() -> dict[str, float]:
    return {
        "PA": 600.0, "IP": 0.0, "R": 90.0, "HR": 25.0, "RBI": 85.0, "SB": 20.0,
        "OPS": 0.820, "W": 0.0, "SV": 0.0, "K": 0.0, "ERA": 0.0, "WHIP": 0.0,
    }


def _pitcher_line() -> dict[str, float]:
    return {
        "PA": 0.0, "IP": 200.0, "R": 0.0, "HR": 0.0, "RBI": 0.0, "SB": 0.0,
        "OPS": 0.0, "W": 15.0, "SV": 0.0, "K": 230.0, "ERA": 3.100,
        "WHIP": 1.050,
    }


def test_vacate_then_readd_is_identity():
    """Removing a player then re-adding him must reproduce the totals exactly.

    If this drifts, every swap comparison is measured against a moving
    baseline.
    """
    totals = _totals()
    for line in (_hitter_line(), _pitcher_line()):
        restored = totals_after_adding(vacate_slot(totals, line), line)
        for key in totals:
            assert restored[key] == pytest.approx(totals[key], abs=1e-9), (
                f"Round trip changed {key}: {totals[key]} -> {restored[key]}. "
                f"The ratio update is not invertible."
            )


def test_ratio_update_is_exact_not_linear():
    """OPS after adding a player must be the true volume-weighted average.

    The first-order form used for screening is my_OPS + PA(OPS_p − my_OPS)/PA_team,
    which drifts as the player's share of team volume grows. A starting pitcher
    is 15-20% of a roto team's innings, so exactness matters here.
    """
    totals = _totals(PA=1000.0, OPS=0.700)
    line = dict(_hitter_line(), PA=1000.0, OPS=0.900)
    updated = totals_after_adding(totals, line)
    assert updated["OPS"] == pytest.approx(0.800), (
        f"Equal volumes at .700 and .900 must average to exactly .800, got "
        f"{updated['OPS']}"
    )
    first_order = 0.700 + 1000.0 * (0.900 - 0.700) / 1000.0
    assert abs(first_order - 0.800) > 0.05, (
        "This fixture is supposed to be a case where the first-order form is "
        "visibly wrong; if it agrees, the test proves nothing."
    )


def test_counting_categories_add_directly():
    totals = _totals()
    updated = totals_after_adding(totals, _hitter_line())
    assert updated["HR"] == pytest.approx(totals["HR"] + 25.0)
    assert updated["PA"] == pytest.approx(totals["PA"] + 600.0)
    assert updated["IP"] == pytest.approx(totals["IP"]), (
        "A hitter must not change team innings."
    )


def test_self_swap_is_exactly_zero():
    """Replacing a player with himself must move nothing.

    Guards both the vacate/re-add algebra and the common-random-numbers
    discipline: with independent seeds this would be Monte Carlo noise instead
    of zero.
    """
    totals, field = _totals(), _field()
    sigmas = {c: 20.0 if c not in ("OPS", "ERA", "WHIP") else 0.02 for c in ALL_CATEGORIES}
    line = _hitter_line()
    delta = swap_delta_p(totals, field, sigmas, line, line, n_sims=_SIMS)
    assert delta == 0.0, (
        f"Self-swap moved p_win by {delta}. Either the round trip is lossy or "
        f"the two simulations used different seeds."
    )


def test_gradient_signs_follow_category_polarity():
    """ERA and WHIP must come out NEGATIVE; everything else positive.

    The simulator handles polarity internally, so the gradient code must NOT
    pre-negate. If it did, a good pitcher would score as a bad one.
    """
    totals, field = _totals(), _field()
    sigmas = {
        "R": 25.0, "HR": 12.0, "RBI": 24.0, "SB": 12.0, "OPS": 0.010,
        "W": 4.0, "SV": 5.0, "K": 30.0, "ERA": 0.15, "WHIP": 0.025,
    }
    gradient, diagnostics = championship_gradient(
        totals, field, sigmas, n_sims=20000
    )
    assert not diagnostics["degenerate"], (
        f"A team at the league mean should be live, but baseline was "
        f"{diagnostics['baseline']}."
    )
    for category in ALL_CATEGORIES:
        if category in NEGATIVE_CATEGORIES:
            assert gradient[category] < 0.0, (
                f"G_{category} = {gradient[category]}; lower is better for it, "
                f"so raising the total must reduce P(win)."
            )
        else:
            assert gradient[category] > 0.0, (
                f"G_{category} = {gradient[category]}; higher is better for it."
            )


def test_degenerate_season_forces_gradient_to_zero():
    """A decided race must report zero leverage, not Monte Carlo noise.

    Handed a hopeless roster, every simulated season is lost, so the finite
    differences are pure noise. Propagating that as signal would have the model
    ranking players on nothing.
    """
    hopeless = _totals(
        R=1.0, HR=1.0, RBI=1.0, SB=1.0, OPS=0.200, W=1.0, SV=0.0, K=10.0,
        ERA=9.0, WHIP=2.5,
    )
    sigmas = {
        "R": 25.0, "HR": 12.0, "RBI": 24.0, "SB": 12.0, "OPS": 0.010,
        "W": 4.0, "SV": 5.0, "K": 30.0, "ERA": 0.15, "WHIP": 0.025,
    }
    gradient, diagnostics = championship_gradient(
        hopeless, _field(), sigmas, n_sims=_SIMS
    )
    assert diagnostics["baseline"] < DEGENERATE_P_FLOOR
    assert diagnostics["degenerate"] is True
    assert all(value == 0.0 for value in gradient.values()), (
        f"Degenerate gradient should be all zeros, got {gradient}"
    )
    assert "raw" in diagnostics, (
        "The unforced gradient must still be reported so a degenerate case can "
        "be inspected rather than merely hidden."
    )


def test_nominal_league_is_symmetric():
    """With the whole field at the mean, p_win must be 1/(teams) by symmetry.

    This is the check that caught the frozen-opponent bug: nominalising only my
    own roster left a 79.5% juggernaut in the field and put a mean team at 2.2%.
    """
    mean = _totals()
    field = nominal_league(mean, 6)
    sigmas = {
        "R": 25.0, "HR": 12.0, "RBI": 24.0, "SB": 12.0, "OPS": 0.010,
        "W": 4.0, "SV": 5.0, "K": 30.0, "ERA": 0.15, "WHIP": 0.025,
    }
    probability = win_probability(mean, field, sigmas, n_sims=40000)
    assert probability == pytest.approx(1.0 / 7.0, abs=0.01), (
        f"Seven identical teams must each win 1/7 of the time, got "
        f"{probability:.4f}."
    )


def test_nominal_totals_interpolates_between_me_and_the_mean():
    mine = _totals(HR=300.0)
    mean = _totals(HR=200.0)
    assert nominal_totals(mine, mean, 0.0, 3)["HR"] == pytest.approx(200.0), (
        "psi=0 must snap fully to the league mean."
    )
    assert nominal_totals(mine, mean, 1.0, 3)["HR"] == pytest.approx(300.0), (
        "psi=1 must hold my own roster forever."
    )
    half = nominal_totals(mine, mean, 0.5, 1)["HR"]
    assert half == pytest.approx(250.0), f"psi=0.5 at t=1 should be 250, got {half}"
    assert nominal_totals(mine, mean, 0.5, 0)["HR"] == pytest.approx(300.0), (
        "At horizon 0 the reference is always my actual roster."
    )
    with pytest.raises(AssertionError, match="psi must lie"):
        nominal_totals(mine, mean, 1.5, 1)


def test_inflate_sigmas_widens_monotonically():
    sigmas = {"R": 20.0, "OPS": 0.01}
    assert inflate_sigmas(sigmas, 0, 0.5)["R"] == pytest.approx(20.0), (
        "Horizon 0 must leave sigma untouched."
    )
    widths = [inflate_sigmas(sigmas, t, 0.5)["R"] for t in range(5)]
    assert all(b > a for a, b in zip(widths, widths[1:])), (
        f"Sigma must grow with horizon; got {widths}. This is the derived "
        f"discount, and a flat sequence removes it."
    )
    with pytest.raises(AssertionError, match="drift_variance"):
        inflate_sigmas(sigmas, 2, -0.1)


def test_score_line_matches_the_mew_formula():
    """The screening score must be exactly gradient . line, ratios differenced."""
    totals = _totals()
    line = _hitter_line()
    gradient = {c: (2.0 if c not in NEGATIVE_CATEGORIES else -2.0) for c in ALL_CATEGORIES}
    expected = 2.0 * (90.0 + 25.0 + 85.0 + 20.0)
    expected += 2.0 * 600.0 * (0.820 - totals["OPS"]) / totals["PA"]
    assert score_line(line, gradient, totals) == pytest.approx(expected), (
        "score_line diverged from the documented MEW formula."
    )


def test_league_mean_totals_averages_every_team_once():
    mine = _totals(HR=300.0)
    field = _field(6, HR=100.0)
    mean = league_mean_totals(mine, field)
    assert mean["HR"] == pytest.approx((300.0 + 6 * 100.0) / 7.0), (
        f"Each team counts once regardless of its volume, got {mean['HR']}"
    )


def test_resolution_floor_scales_with_draws():
    assert resolution_floor(50000) < resolution_floor(5000), (
        "More draws must resolve smaller differences."
    )


def test_exact_delta_p_beats_linear_when_convex():
    """Below the cutoff, exact delta-P must EXCEED the linear estimate.

    P is convex there, so the first-order score understates. If this inverts,
    either the convexity reasoning or the gradient's sign handling is wrong.
    """
    # Just below the middle of a graded field: live, but on the convex side.
    behind = _totals(
        R=760.0, HR=208.0, RBI=740.0, SB=112.0, OPS=0.726, W=66.0, SV=57.0,
        K=1240.0, ERA=4.05, WHIP=1.30,
    )
    field = _graded_field()
    sigmas = {
        "R": 40.0, "HR": 18.0, "RBI": 38.0, "SB": 18.0, "OPS": 0.020,
        "W": 6.0, "SV": 7.0, "K": 60.0, "ERA": 0.28, "WHIP": 0.045,
    }
    gradient, diagnostics = championship_gradient(
        behind, field, sigmas, n_sims=60000
    )
    assert not diagnostics["degenerate"], (
        f"Fixture landed in the degenerate branch (baseline "
        f"{diagnostics['baseline']}), so it measures nothing. The team must be "
        f"behind but live."
    )
    line = _hitter_line()
    linear = score_line(line, gradient, behind)
    exact = exact_delta_p(
        behind, field, sigmas, line, baseline=diagnostics["baseline"], n_sims=60000
    )
    assert exact > 0.0 and linear > 0.0, f"linear={linear} exact={exact}"
    assert exact > linear, (
        f"Exact {exact:.5f} should exceed linear {linear:.5f} for a team below "
        f"the cutoff, where P is convex."
    )


def test_build_season_context_shapes_and_symmetry():
    totals, field = _totals(), _field()
    sigmas = {
        "R": 25.0, "HR": 12.0, "RBI": 24.0, "SB": 12.0, "OPS": 0.010,
        "W": 4.0, "SV": 5.0, "K": 30.0, "ERA": 0.15, "WHIP": 0.025,
    }
    context = build_season_context(totals, field, sigmas, 4, n_sims=20000)
    for key in ("gradients", "references", "fields", "sigmas", "baselines"):
        assert len(context[key]) == 4, f"{key} has {len(context[key])} entries, want 4"
    assert context["baselines"][1] == pytest.approx(1.0 / 7.0, abs=0.02), (
        f"Nominal future seasons must sit at 1/7; got {context['baselines'][1]}"
    )
    magnitudes = [abs(context["gradients"][t]["HR"]) for t in range(1, 4)]
    assert all(b < a for a, b in zip(magnitudes, magnitudes[1:])), (
        f"Gradient magnitude must fall with horizon from sigma inflation alone; "
        f"got {magnitudes}. That decline IS the derived discount, separate "
        f"from beta."
    )
