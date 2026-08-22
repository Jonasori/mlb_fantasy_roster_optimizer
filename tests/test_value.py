"""
Offline tests for the dynasty objective V(p; beta).
No network. Per AGENTS.md: no classes, no fixtures, no mocking.

The load-bearing guards here:

  test_project_line_skips_opposite_type_categories
      The silver table fills the other player type's categories with 0.0, so a
      hitter's line carries W/SV/K. There is no hitter decay curve for W, and
      looking one up is a crash.
  test_counting_stats_take_both_decays
      A count is a rate times playing time. Applying only the rate curve is how
      a model overvalues speed -- stolen-base rate is near-flat from 22 to 26
      while plate-appearance attrition is steepest for exactly those profiles.
  test_annualize_scales_volume_not_rates
      Season 0 is the season's remainder; later seasons are whole. In August
      that is a 5x error on every future season if missed.
  test_breakeven_reports_multiple_crossings_honestly
      Two players can cross more than once. A single break-even number would
      then be a lie, and Descartes' rule is what tells us which case we are in.
"""

import numpy as np
import pytest

from optimizer.value import (
    annualize,
    beta_sweep,
    branch_payoffs,
    breakeven_beta,
    dominates,
    net_value,
    pareto_frontier,
    player_value,
    project_line,
    single_branch,
)


def _line(**overrides) -> dict[str, float]:
    base = {
        "PA": 600.0, "IP": 0.0, "R": 90.0, "HR": 25.0, "RBI": 85.0, "SB": 20.0,
        "OPS": 0.820, "W": 0.0, "SV": 0.0, "K": 0.0, "ERA": 0.0, "WHIP": 0.0,
    }
    base.update(overrides)
    return base


def _flat_decay(_age, _years, _category, _role) -> float:
    return 1.0


def _flat_survival(_age, _years, _role) -> float:
    return 1.0


def _halving_decay(_age, years, _category, _role) -> float:
    return 0.5**years


def _totals() -> dict[str, float]:
    return {
        "R": 800.0, "HR": 220.0, "RBI": 780.0, "SB": 120.0, "OPS": 0.740,
        "W": 70.0, "SV": 60.0, "K": 1300.0, "ERA": 3.900, "WHIP": 1.250,
        "PA": 6500.0, "IP": 1300.0,
    }


def _unit_gradient() -> dict[str, float]:
    return {
        "R": 1.0, "HR": 1.0, "RBI": 1.0, "SB": 1.0, "OPS": 1.0,
        "W": 1.0, "SV": 1.0, "K": 1.0, "ERA": -1.0, "WHIP": -1.0,
    }


def test_project_line_horizon_zero_is_unchanged():
    line = _line()
    assert project_line(line, 27, 0, "hitter", _halving_decay, _flat_survival) == line


def test_project_line_skips_opposite_type_categories():
    """A hitter's zero-valued W/SV/K must not trigger a pitcher curve lookup."""

    def exploding_decay(_age, _years, category, role):
        assert not (role == "hitter" and category in ("W", "SV", "K")), (
            f"Looked up a {category} curve for a hitter. Zero-valued "
            f"opposite-type categories must be skipped, not queried."
        )
        return 0.9

    projected = project_line(_line(), 27, 3, "hitter", exploding_decay, _flat_survival)
    assert projected["W"] == 0.0 and projected["K"] == 0.0


def test_counting_stats_take_both_decays():
    """A count decays by its rate curve AND the volume curve; a rate does not."""
    projected = project_line(
        _line(), 27, 1, "hitter", _halving_decay, _flat_survival
    )
    assert projected["PA"] == pytest.approx(600.0 * 0.5), (
        f"Volume takes one factor, got {projected['PA']}"
    )
    assert projected["HR"] == pytest.approx(25.0 * 0.5 * 0.5), (
        f"A count takes rate AND volume, so 0.25x here; got {projected['HR']}. "
        f"Applying only the rate curve overvalues volume-fragile profiles."
    )
    assert projected["OPS"] == pytest.approx(0.820 * 0.5), (
        f"A rate takes only its own curve, got {projected['OPS']}"
    )


def test_survival_scales_volume_but_never_the_rate():
    """Survival must not degrade a rate.

    A player who does not play has PA = 0, which zeroes his ratio contribution
    on its own. He does not acquire a worse OPS. Multiplying the rate as well
    applies the haircut twice.
    """

    def half_survival(_age, _years, _role) -> float:
        return 0.5

    projected = project_line(_line(), 27, 1, "hitter", _flat_decay, half_survival)
    assert projected["PA"] == pytest.approx(300.0)
    assert projected["HR"] == pytest.approx(12.5)
    assert projected["OPS"] == pytest.approx(0.820), (
        f"OPS must be untouched by survival, got {projected['OPS']}"
    )


def test_project_line_zeroes_past_the_fitted_band():
    """A zero decay factor must zero the whole line, not scale part of it."""

    def dead(_age, _years, _category, _role) -> float:
        return 0.0

    projected = project_line(_line(), 39, 5, "hitter", dead, _flat_survival)
    assert all(value == 0.0 for value in projected.values()), (
        f"Past the band every entry must be zero, got {projected}"
    )


def test_annualize_scales_volume_not_rates():
    annual = annualize(_line(), 0.20)
    assert annual["PA"] == pytest.approx(3000.0), (
        f"600 PA over a fifth of a season annualizes to 3000, got {annual['PA']}"
    )
    assert annual["HR"] == pytest.approx(125.0)
    assert annual["OPS"] == pytest.approx(0.820), (
        f"A rate must not be scaled by the horizon, got {annual['OPS']}"
    )
    with pytest.raises(AssertionError, match="season_fraction_remaining"):
        annualize(_line(), 0.0)


def test_single_branch_differs_only_in_horizon():
    now, later = single_branch(_line(), 27, "hitter", 0.20)
    assert now[0]["line"]["PA"] == pytest.approx(600.0)
    assert later[0]["line"]["PA"] == pytest.approx(3000.0)
    assert now[0]["prob"] == 1.0 and later[0]["prob"] == 1.0
    assert now[0]["arrive"] == 0 and now[0]["arrive_age"] == 27


def test_branch_payoffs_requires_probabilities_to_sum_to_one():
    """Missing mass silently discounts a player; it must be an error.

    The never-arrives branch has to be present explicitly as an all-zero line.
    """
    branches = [
        {"prob": 0.4, "line": _line(), "arrive": 0, "arrive_age": 27,
         "role": "hitter"}
    ]
    with pytest.raises(AssertionError, match="sum to"):
        branch_payoffs(
            branches, [_unit_gradient()], [_totals()], _flat_decay,
            _flat_survival, 1,
        )


def test_branch_payoffs_respects_arrival_season():
    """A prospect contributes exactly zero before he arrives."""
    branches = [
        {"prob": 1.0, "line": _line(), "arrive": 3, "arrive_age": 22,
         "role": "hitter"}
    ]
    payoffs = branch_payoffs(
        branches, [_unit_gradient()] * 6, [_totals()] * 6, _flat_decay,
        _flat_survival, 6,
    )
    assert list(payoffs[:3]) == [0.0, 0.0, 0.0], (
        f"Seasons before arrival must be zero, got {payoffs[:3]}"
    )
    assert payoffs[3] > 0.0, "The arrival season must pay."


def test_branch_payoffs_is_a_probability_weighted_mixture():
    star = _line(HR=50.0)
    bust = {key: 0.0 for key in _line()}
    mixed = branch_payoffs(
        [
            {"prob": 0.25, "line": star, "arrive": 0, "arrive_age": 24,
             "role": "hitter"},
            {"prob": 0.75, "line": bust, "arrive": 0, "arrive_age": 24,
             "role": "hitter"},
        ],
        [_unit_gradient()], [_totals()], _flat_decay, _flat_survival, 1,
    )
    pure = branch_payoffs(
        [{"prob": 1.0, "line": star, "arrive": 0, "arrive_age": 24,
          "role": "hitter"}],
        [_unit_gradient()], [_totals()], _flat_decay, _flat_survival, 1,
    )
    assert mixed[0] == pytest.approx(0.25 * pure[0]), (
        f"A 25% star branch must pay a quarter of the star, got {mixed[0]} vs "
        f"{0.25 * pure[0]}"
    )


def test_player_value_discounts_and_converges():
    payoffs = np.array([1.0, 1.0, 1.0, 1.0])
    assert player_value(payoffs, 1.0) == pytest.approx(4.0)
    assert player_value(payoffs, 0.5) == pytest.approx(1 + 0.5 + 0.25 + 0.125)
    with pytest.raises(AssertionError, match="beta must lie"):
        player_value(payoffs, 1.5)


def test_player_value_terminates_on_a_decaying_series_at_beta_one():
    """beta = 1 must be finite: the data supplies its own discount."""
    payoffs = np.array([0.5**t for t in range(30)])
    value = player_value(payoffs, 1.0)
    assert value == pytest.approx(2.0, abs=1e-6), (
        f"sum of 0.5^t from t=0 is 2.0, not 1.0; got {value}"
    )


def test_dominates_is_parameter_free():
    better = np.array([2.0, 2.0, 2.0])
    worse = np.array([1.0, 1.0, 1.0])
    crossing = np.array([3.0, 0.5, 0.5])
    assert dominates(better, worse)
    assert not dominates(worse, better)
    assert not dominates(crossing, better), (
        "A player who leads early and trails late does not dominate."
    )
    assert not dominates(better, better), (
        "Domination must be strict somewhere, so a tie is not domination."
    )


def test_breakeven_finds_the_unique_prospect_versus_veteran_crossing():
    """One sign change means exactly one crossing, so a single beta is honest."""
    veteran = np.array([1.0, 0.8, 0.2, 0.0, 0.0])
    prospect = np.array([0.0, 0.0, 0.6, 0.9, 1.0])
    result = breakeven_beta(veteran, prospect)
    assert result["sign_changes"] == 1, (
        f"Expected one sign change for prospect-vs-veteran, got "
        f"{result['sign_changes']}"
    )
    assert result["unique"] is True
    assert len(result["roots"]) == 1
    root = result["roots"][0]
    assert 0.0 < root < 1.0, f"Crossing should be interior, got {root}"
    # Verify it really is the crossing.
    assert player_value(veteran, root * 0.9) > player_value(prospect, root * 0.9)
    assert player_value(veteran, min(root * 1.1, 1.0)) < player_value(
        prospect, min(root * 1.1, 1.0)
    )


def test_breakeven_reports_multiple_crossings_honestly():
    a = np.array([1.0, -3.0, 3.0])
    b = np.array([0.0, 0.0, 0.0])
    result = breakeven_beta(a, b)
    assert result["sign_changes"] >= 2, (
        f"This pair alternates sign; got {result['sign_changes']}"
    )
    assert result["unique"] is False, (
        "With more than one sign change a single break-even beta would "
        "misrepresent the pair, and the report must say so."
    )
    assert "misrepresent" in result["summary"] or len(result["roots"]) != 1


def test_breakeven_handles_no_crossing():
    result = breakeven_beta(np.array([2.0, 2.0]), np.array([1.0, 1.0]))
    assert result["roots"] == []
    assert "every beta" in result["summary"]


def test_breakeven_handles_identical_players():
    result = breakeven_beta(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert result["roots"] == []
    assert result["sign_changes"] == 0


def test_net_value_goes_negative_against_a_better_alternative():
    """The 'mid prospect' arithmetic: worth less than the slot's alternative."""
    mid_prospect = np.array([0.0, 0.0, 0.05, 0.05, 0.05])
    alternative = np.array([0.0, 0.0, 0.20, 0.20, 0.20])
    assert net_value(mid_prospect, alternative, 0.9) < 0.0, (
        "A prospect who matures into less than the slot's best alternative must "
        "carry negative net value. This is the doctrine as arithmetic."
    )
    assert net_value(alternative, mid_prospect, 0.9) > 0.0


def test_pareto_frontier_drops_only_dominated_players():
    matrix = {
        "star": np.array([2.0, 2.0, 2.0]),
        "veteran": np.array([3.0, 0.5, 0.1]),
        "prospect": np.array([0.0, 1.0, 3.0]),
        "dominated": np.array([1.0, 1.0, 1.0]),
    }
    frontier = pareto_frontier(matrix)
    assert "dominated" not in frontier, (
        "A player worse in every season than 'star' is optimal at no beta."
    )
    for name in ("star", "veteran", "prospect"):
        assert name in frontier, f"{name} should survive the frontier"


def test_beta_sweep_flags_rank_instability():
    # The veteran must actually be overtakeable: front-loaded but not so large
    # that no beta in (0, 1] can catch him.
    matrix = {
        "veteran": np.array([1.0, 0.3, 0.05]),
        "prospect": np.array([0.0, 0.6, 1.6]),
        "steady": np.array([0.5, 0.5, 0.5]),
    }
    sweep = beta_sweep(matrix, (0.2, 0.5, 0.9))
    assert sweep.loc["veteran", "rank_swing"] > 0, (
        "The veteran must lead at low beta and trail at high beta, so its rank "
        "has to move across the sweep."
    )
    assert sweep.loc["veteran", "V@0.2"] > sweep.loc["prospect", "V@0.2"]
    assert sweep.loc["veteran", "V@0.9"] < sweep.loc["prospect", "V@0.9"]
