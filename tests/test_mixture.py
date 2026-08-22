"""
Offline tests for outcome mixtures.
No network. Per AGENTS.md: no classes, no fixtures, no mocking.

Each of these guards a failure that actually occurred while building the module:

  test_backoff_preserves_the_performance_signal
      Dropping straight from the full cell to (age, level) made five different
      22-year-old Double-A hitters score IDENTICALLY across the <90 to 125+
      performance range. Useless for choosing among prospects.
  test_shrinkage_pulls_a_thin_cell_off_certainty
      A thin cell gave 19-year-olds in Double-A P(never reaches MLB) = 0.000 --
      certainty of arrival, off a handful of observations, for exactly the
      rarest and most exciting players on the board.
  test_branch_probabilities_include_the_never_mass
      Missing mass silently discounts every prospect.
"""

import numpy as np
import pandas as pd
import pytest

from optimizer.mixture import (
    LINE_STATS,
    MAX_ARRIVAL,
    joint_outcome_table,
    major_leaguer_branches,
    prospect_branches,
    tier_archetype_lines,
)


def _cohort(rows: list[dict]) -> pd.DataFrame:
    """Synthetic cohort with the columns the builders require."""
    frame = pd.DataFrame(rows)
    defaults = {
        "player_type": "hitter", "age": 21, "sport_id": 12,
        "age_rel_bucket": "-2..-1", "perf_bucket": "110-125",
        "tier": "fringe", "years_to_mlb": 2.0, "n_mlb_seasons": 3.0,
        "car_PA": 1200.0, "car_IP": 0.0, "car_R": 150.0, "car_HR": 40.0,
        "car_RBI": 140.0, "car_SB": 20.0, "car_W": 0.0, "car_SV": 0.0,
        "car_K": 0.0, "car_OPS": 0.700, "car_ERA": 0.0, "car_WHIP": 0.0,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    if "player_id" not in frame.columns:
        frame["player_id"] = range(len(frame))
    return frame


def _archetypes() -> pd.DataFrame:
    def line(**kw):
        base = {stat: 0.0 for stat in LINE_STATS}
        base.update(kw)
        return base

    return pd.DataFrame(
        [
            {"player_type": "hitter", "tier": "never", "n_players": 100, **line()},
            {"player_type": "hitter", "tier": "fringe", "n_players": 50,
             **line(PA=600.0, R=66.0, HR=14.0, RBI=60.0, SB=9.0, OPS=0.634)},
            {"player_type": "hitter", "tier": "regular", "n_players": 20,
             **line(PA=600.0, R=77.0, HR=20.0, RBI=73.0, SB=11.0, OPS=0.780)},
            {"player_type": "hitter", "tier": "star", "n_players": 10,
             **line(PA=600.0, R=84.0, HR=25.0, RBI=79.0, SB=14.0, OPS=0.832)},
        ]
    )


def test_archetype_lines_are_ordered_by_tier():
    rows = []
    for index, (tier, ops, hr) in enumerate(
        (("fringe", 0.630, 12.0), ("regular", 0.780, 20.0), ("star", 0.840, 30.0))
    ):
        for player in range(10):
            rows.append(
                {"player_id": index * 100 + player, "tier": tier,
                 "car_OPS": ops, "car_HR": hr * 3, "n_mlb_seasons": 3.0}
            )
    lines = tier_archetype_lines(_cohort(rows))
    hitters = lines[lines["player_type"] == "hitter"].set_index("tier")
    assert hitters.loc["never", "PA"] == 0.0, (
        "The never tier must be an all-zero line."
    )
    assert (
        hitters.loc["star", "OPS"]
        > hitters.loc["regular", "OPS"]
        > hitters.loc["fringe", "OPS"]
    ), f"Tier OPS is not ordered: {hitters['OPS'].to_dict()}"
    assert hitters.loc["star", "PA"] == pytest.approx(600.0), (
        "Lines must be scaled to a full season's volume."
    )


def test_archetype_divides_by_seasons_played_not_the_window():
    """A line is what a season LOOKS LIKE, not a career averaged over 8 years.

    Survival in value.project_line already prices whether he plays; baking
    absence into the line as well applies the discount twice.
    """
    rows = [
        {"player_id": i, "tier": "star", "n_mlb_seasons": 2.0, "car_PA": 1200.0,
         "car_HR": 60.0, "car_OPS": 0.840}
        for i in range(10)
    ]
    lines = tier_archetype_lines(_cohort(rows))
    star = lines[(lines.player_type == "hitter") & (lines.tier == "star")].iloc[0]
    # 60 HR over 2 seasons = 30/season at 600 PA/season, already full-season.
    assert star["HR"] == pytest.approx(30.0), (
        f"Expected 30 HR per season (60 over 2), got {star['HR']}. Dividing by "
        f"the 8-year window instead would give 7.5."
    )


def test_joint_table_normalizes_within_each_state():
    rows = []
    for index in range(40):
        rows.append(
            {"player_id": index, "tier": "star" if index < 4 else "fringe",
             "years_to_mlb": 2.0}
        )
    joint = joint_outcome_table(_cohort(rows))
    for conditioning in ("full", "age_level_perf", "age_level"):
        block = joint[joint["conditioning"] == conditioning]
        assert block["p"].sum() == pytest.approx(1.0), (
            f"{conditioning} probabilities sum to {block['p'].sum()}, not 1.0"
        )


def test_never_tier_is_forced_to_the_absorbing_arrival_state():
    rows = [
        {"player_id": i, "tier": "never", "years_to_mlb": np.nan} for i in range(30)
    ]
    joint = joint_outcome_table(_cohort(rows))
    never = joint[joint["tier"] == "never"]
    assert (never["arrive"] == -1).all(), (
        f"A never-reached player cannot have an arrival year; got "
        f"{never['arrive'].unique()}"
    )


def test_arrival_beyond_the_window_becomes_never():
    rows = [
        {"player_id": i, "tier": "fringe", "years_to_mlb": float(MAX_ARRIVAL + 3)}
        for i in range(30)
    ]
    joint = joint_outcome_table(_cohort(rows))
    assert (joint["arrive"] == -1).all(), (
        f"An arrival past the {MAX_ARRIVAL}-year observed window is unmeasured "
        f"and must fold into the absorbing state; got {joint['arrive'].unique()}"
    )


def test_backoff_preserves_the_performance_signal():
    """Two players differing ONLY in performance must not score identically.

    The (age, level) rung drops performance. The intermediate rung keeps it,
    which is the whole reason that rung exists.
    """
    rows = []
    # A thin fully-conditioned cell, but a well-populated (age, level, perf) one.
    for index in range(60):
        strong = index < 30
        rows.append(
            {
                "player_id": index,
                "perf_bucket": "125+" if strong else "<90",
                "age_rel_bucket": f"bucket_{index}",  # keeps every full cell at n=1
                "tier": "star" if strong else "never",
                "years_to_mlb": 1.0 if strong else np.nan,
            }
        )
    joint = joint_outcome_table(_cohort(rows))
    archetypes = _archetypes()

    strong_branches, strong_cond = prospect_branches(
        joint, archetypes, "hitter", 21, 12, "bucket_0", "125+", "hitter"
    )
    weak_branches, weak_cond = prospect_branches(
        joint, archetypes, "hitter", 21, 12, "bucket_59", "<90", "hitter"
    )
    assert strong_cond == "age_level_perf" and weak_cond == "age_level_perf", (
        f"Expected the performance-preserving rung, got {strong_cond} / "
        f"{weak_cond}. A thin full cell must not fall all the way through."
    )

    def star_mass(branches):
        return sum(
            b["prob"] for b in branches if b["line"].get("OPS", 0.0) > 0.82
        )

    assert star_mass(strong_branches) > star_mass(weak_branches), (
        f"The 125+ performer got star mass {star_mass(strong_branches):.3f} "
        f"against the <90 performer's {star_mass(weak_branches):.3f}. The "
        f"backoff has discarded the performance signal."
    )


def test_shrinkage_pulls_a_thin_cell_off_certainty():
    """A 3-observation cell must not assert certainty of arrival."""
    rows = []
    # Level marginal: mostly never reaches.
    for index in range(200):
        rows.append(
            {"player_id": index, "age": 24, "tier": "never",
             "years_to_mlb": np.nan, "perf_bucket": "100-110"}
        )
    # A thin cell at a different age where everyone arrived.
    for index in range(3):
        rows.append(
            {"player_id": 1000 + index, "age": 19, "tier": "star",
             "years_to_mlb": 1.0, "perf_bucket": "125+"}
        )
    joint = joint_outcome_table(_cohort(rows))
    branches, _ = prospect_branches(
        joint, _archetypes(), "hitter", 19, 12, "-2..-1", "125+", "hitter"
    )
    never_mass = sum(
        b["prob"] for b in branches
        if all(value == 0.0 for value in b["line"].values())
    )
    assert never_mass > 0.05, (
        f"A three-observation cell produced P(never) = {never_mass:.4f}. "
        f"Shrinkage toward the level marginal must prevent a thin cell from "
        f"claiming certainty — that is how the rarest prospects get overvalued."
    )
    assert never_mass < 1.0


def test_branch_probabilities_include_the_never_mass():
    rows = []
    for index in range(60):
        reached = index < 20
        rows.append(
            {"player_id": index, "tier": "fringe" if reached else "never",
             "years_to_mlb": 2.0 if reached else np.nan}
        )
    joint = joint_outcome_table(_cohort(rows))
    branches, _ = prospect_branches(
        joint, _archetypes(), "hitter", 21, 12, "-2..-1", "110-125", "hitter"
    )
    total = sum(b["prob"] for b in branches)
    assert total == pytest.approx(1.0), (
        f"Branch probabilities sum to {total:.6f}. Missing mass silently "
        f"discounts the whole player."
    )
    zero_lines = [
        b for b in branches if all(v == 0.0 for v in b["line"].values())
    ]
    assert zero_lines, "The never-arrives branch must be present explicitly."


def test_prospect_is_aged_from_arrival_not_from_today():
    """A teenager must never be aged from his current age.

    The decay curves do not exist below 20, so aging a 17-year-old from today
    extrapolates a curve off data that cannot exist.
    """
    rows = [
        {"player_id": i, "age": 17, "sport_id": 16, "tier": "star",
         "years_to_mlb": 5.0}
        for i in range(40)
    ]
    joint = joint_outcome_table(_cohort(rows))
    branches, _ = prospect_branches(
        joint, _archetypes(), "hitter", 17, 16, "-2..-1", "110-125", "hitter"
    )
    arriving = [b for b in branches if b["arrive"] > 0]
    assert arriving, "Expected at least one arriving branch."
    for branch in branches:
        assert branch["arrive_age"] >= 20, (
            f"arrive_age {branch['arrive_age']} is below the fitted band. A "
            f"prospect must be aged from his projected ARRIVAL age."
        )


def test_unsupported_state_raises_rather_than_defaulting():
    rows = [{"player_id": i, "age": 21, "sport_id": 12} for i in range(40)]
    joint = joint_outcome_table(_cohort(rows))
    with pytest.raises(AssertionError, match="no cohort support"):
        prospect_branches(
            joint, _archetypes(), "hitter", 21, 99, "-2..-1", "110-125", "hitter"
        )


def test_major_leaguer_mixture_is_one_certain_branch():
    line = {stat: 0.0 for stat in LINE_STATS}
    line.update({"PA": 120.0, "HR": 5.0, "OPS": 0.800})
    now, later = major_leaguer_branches(line, 27, "hitter", 0.20)
    assert len(now) == len(later) == 1
    assert now[0]["prob"] == 1.0
    assert later[0]["line"]["PA"] == pytest.approx(600.0), (
        "Later seasons must use the annualized line."
    )
    assert later[0]["line"]["OPS"] == pytest.approx(0.800), (
        "Annualizing must not touch a rate."
    )
