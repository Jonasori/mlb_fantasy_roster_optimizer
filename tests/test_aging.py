"""
Offline tests for the aging/attrition priors.
No network. Per AGENTS.md: no classes, no fixtures, no mocking.

These test the METHOD on synthetic populations where the right answer is known
by construction, not the fitted numbers. The four method choices in
`data_prep.aging`'s docstring each get a test, because each one silently
changes the answer if it regresses:

  pooled ratio      a few extreme small-count ratios must not dominate a cell
  league-centering  a league-wide shift must not read as aging
  harmonic weights  a short season must not carry a full one's weight
  censoring         an unplayed future season must not count as a failure
"""

import numpy as np
import pandas as pd
import pytest

from data_prep.aging import (
    LAST_COMPLETE_SEASON,
    MAX_AGE,
    MIN_AGE,
    _harmonic_weight,
    _assign_deciles,
    build_decay_table,
    build_survival_table,
    cumulative_decay,
    prepare_hitters,
    prepare_pitchers,
    survival_factor,
)


def _hitter_seasons(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic prepared-hitter frame. `rows` carry the fields set."""
    frame = pd.DataFrame(rows)
    for column, default in (
        ("VOL", 600.0),
        ("R", 80.0),
        ("HR", 20.0),
        ("RBI", 80.0),
        ("SB", 10.0),
        ("OPS", 0.750),
    ):
        if column not in frame.columns:
            frame[column] = default
    frame["role"] = "hitter"
    return frame


def test_harmonic_weight():
    weight = _harmonic_weight(
        pd.Series([600.0, 100.0, 0.0]), pd.Series([600.0, 600.0, 600.0])
    )
    assert weight[0] == pytest.approx(600.0), (
        f"Equal volumes should weight at that volume, got {weight[0]}"
    )
    assert weight[1] == pytest.approx(2 * 100 * 600 / 700), (
        f"Harmonic mean of 100 and 600 is wrong: {weight[1]}"
    )
    assert weight[2] == 0.0, (
        "A zero-volume season must carry zero weight, not NaN — NaN would "
        f"propagate into every pooled sum. Got {weight[2]}"
    )


def test_flat_population_has_no_decay():
    """A population that never changes must yield factors of exactly 1.0."""
    rows = []
    for player in range(40):
        for season in (2015, 2016, 2017):
            rows.append({"playerId": player, "season": season, "age": 27 + season - 2015})
    table = build_decay_table(_hitter_seasons(rows), ("OPS", "HR", "VOL"))
    factors = table[table["variant"] == "inclusive"]["factor"]
    assert np.allclose(factors, 1.0), (
        f"A stationary population produced decay: {factors.unique()[:5]}. The "
        f"pooled ratio or the league-centering is introducing drift."
    )


def test_league_wide_shift_is_not_aging():
    """Everyone's HR doubling league-wide must read as zero aging.

    This is the 2023 stolen-base rules change in miniature: league SB volume
    rose 41% on a rule, and an uncentered curve reads that as 27-year-olds
    rediscovering speed.
    """
    rows = []
    for player in range(40):
        rows.append({"playerId": player, "season": 2015, "age": 27, "HR": 20.0})
        rows.append({"playerId": player, "season": 2016, "age": 28, "HR": 40.0})
    table = build_decay_table(_hitter_seasons(rows), ("HR",))
    factors = table[table["variant"] == "inclusive"]["factor"]
    assert np.allclose(factors, 1.0), (
        f"A league-wide doubling leaked into the aging curve: {factors.tolist()}. "
        f"Rates must be divided by their own season's league rate first."
    )


def test_pooled_ratio_resists_small_count_blowup():
    """One 1-to-5 stolen base jump must not move a cell built on regulars.

    `mean(y/x)` would read that single player as a 5.0x observation and drag
    the cell up by ~13%. The pooled `sum(y)/sum(x)` form cannot.
    """
    rows = []
    for player in range(30):
        rows.append({"playerId": player, "season": 2015, "age": 27, "SB": 20.0})
        rows.append({"playerId": player, "season": 2016, "age": 28, "SB": 20.0})
    rows.append({"playerId": 999, "season": 2015, "age": 27, "SB": 1.0})
    rows.append({"playerId": 999, "season": 2016, "age": 28, "SB": 5.0})

    table = build_decay_table(_hitter_seasons(rows), ("SB",))
    factor = float(
        table[(table["variant"] == "inclusive") & (table["age"] == 27)]["factor"].iloc[0]
    )
    naive = (30 * 1.0 + 5.0) / 31
    assert factor < 1.02, (
        f"One small-count outlier moved the cell to {factor:.4f}; the pooled "
        f"ratio should hold near 1.0. A mean-of-ratios would give ~{naive:.3f}."
    )


def test_excluded_season_pairs_are_dropped():
    """No pair may span 2020: a 60-game season against a 162-game one."""
    rows = []
    for player in range(30):
        for season, age in ((2019, 27), (2020, 28), (2021, 29)):
            rows.append({"playerId": player, "season": season, "age": age})
    table = build_decay_table(_hitter_seasons(rows), ("OPS",))
    assert table.empty or not table["age"].isin([27, 28]).any(), (
        f"Pairs touching 2020 survived: ages {sorted(table['age'].unique())}. "
        f"2019->2020 and 2020->2021 must both be dropped."
    )


def test_partial_current_season_is_not_a_base_or_outcome():
    """A pair ending after LAST_COMPLETE_SEASON must be dropped.

    2026 is in progress. Counting it as an outcome year reports every player
    in baseball as having lost playing time.
    """
    rows = []
    for player in range(30):
        rows.append(
            {"playerId": player, "season": LAST_COMPLETE_SEASON, "age": 27, "VOL": 600.0}
        )
        rows.append(
            {
                "playerId": player,
                "season": LAST_COMPLETE_SEASON + 1,
                "age": 28,
                "VOL": 100.0,
            }
        )
    table = build_decay_table(_hitter_seasons(rows), ("VOL",))
    assert table.empty, (
        f"A pair ending in the in-progress season produced {len(table)} cells "
        f"with factors {table['factor'].tolist()[:3]}. It must be dropped."
    )


def test_strict_and_inclusive_differ_on_a_collapse():
    """The survivorship variants must actually diverge.

    Half the population collapses to 200 PA. Strict drops them (both seasons
    must qualify); inclusive keeps them. If the two variants agree, the
    survivorship correction is not being computed at all.
    """
    rows = []
    for player in range(30):
        rows.append({"playerId": player, "season": 2015, "age": 27, "VOL": 600.0})
        collapses = player % 2 == 0
        rows.append(
            {
                "playerId": player,
                "season": 2016,
                "age": 28,
                "VOL": 200.0 if collapses else 600.0,
            }
        )
    table = build_decay_table(_hitter_seasons(rows), ("VOL",))
    strict = table[table["variant"] == "strict"]["factor"].iloc[0]
    inclusive = table[table["variant"] == "inclusive"]["factor"].iloc[0]
    assert strict == pytest.approx(1.0), (
        f"Strict should see only the survivors and find no decline, got {strict}"
    )
    assert inclusive < 0.95, (
        f"Inclusive should see the collapses, got {inclusive}. Identical "
        f"variants mean the survivorship correction is inert."
    )


def test_survival_censoring_excludes_unplayed_seasons():
    """A cohort whose t+k has not happened must leave the DENOMINATOR.

    Without this, recent cohorts count as failures at large k purely because
    the seasons do not exist, deflating every long-horizon rate.
    """
    rows = []
    for player in range(20):
        rows.append(
            {
                "playerId": player,
                "season": LAST_COMPLETE_SEASON,
                "age": 27,
                "VOL": 600.0,
            }
        )
    table = build_survival_table(_hitter_seasons(rows), 500.0, 400.0, max_k=3)
    assert table.empty, (
        f"A cohort based in the last complete season has no observable "
        f"future, so every k must be censored out. Got {len(table)} cells with "
        f"rates {table['rate'].tolist()[:3]} — those are fake zeros."
    )


def test_survival_counts_only_observed_outcomes():
    rows = []
    for player in range(20):
        rows.append({"playerId": player, "season": 2015, "age": 27, "VOL": 600.0})
        # Half hold their job in 2016, half drop below the later floor.
        rows.append(
            {
                "playerId": player,
                "season": 2016,
                "age": 28,
                "VOL": 600.0 if player % 2 == 0 else 100.0,
            }
        )
    table = build_survival_table(_hitter_seasons(rows), 500.0, 400.0, max_k=1)
    row = table[(table["age_band"] == "27-29") & (table["k"] == 1)].iloc[0]
    assert row["n"] == 20, f"Denominator should be all 20 base seasons, got {row['n']}"
    assert row["rate"] == pytest.approx(0.5), (
        f"Half survived, so the rate must be 0.5, got {row['rate']}"
    )


def test_decile_inverts_for_lower_is_better():
    """Decile 9 must be the BEST regardless of category polarity.

    ERA is lower-is-better. If the ranking is not inverted, decile 9 collects
    the worst pitchers and every stratified read is upside down.
    """
    frame = pd.DataFrame(
        {
            "playerId": range(40),
            "season": 2015,
            "age": 27,
            "VOL": 200.0,
            "games": 32,
            "role": "starter",
            "ERA": np.linspace(2.0, 6.0, 40),
        }
    )
    deciles = _assign_deciles(frame, "ERA")
    best_era = frame.loc[deciles.idxmax(), "ERA"]
    worst_era = frame.loc[deciles.idxmin(), "ERA"]
    assert best_era < worst_era, (
        f"Decile 9 has ERA {best_era} and decile 0 has {worst_era}; for a "
        f"lower-is-better category the top decile must hold the LOWER ERA."
    )


def test_prepare_hitters_computes_ops_from_counts():
    """OPS must come from counting columns, not the string rate field.

    StatsAPI writes rates as strings with '.---' for missing, so parsing them
    invites silent coercion. Counts are exact.
    """
    raw = pd.DataFrame(
        {
            "playerId": [1],
            "season": [2015],
            "age": [27],
            "plateAppearances": [700],
            "atBats": [600],
            "hits": [180],
            "baseOnBalls": [90],
            "hitByPitch": [5],
            "sacFlies": [5],
            "totalBases": [300],
            "runs": [100],
            "homeRuns": [30],
            "rbi": [100],
            "stolenBases": [20],
            "ops": [".---"],
        }
    )
    prepared = prepare_hitters(raw)
    expected = (180 + 90 + 5) / (600 + 90 + 5 + 5) + 300 / 600
    assert float(prepared["OPS"].iloc[0]) == pytest.approx(expected), (
        f"OPS should be {expected:.4f} from counts, got "
        f"{float(prepared['OPS'].iloc[0]):.4f} — the string 'ops' column was "
        f"probably read instead."
    )


def test_prepare_pitchers_uses_outs_not_innings_string():
    """IP must come from outs. 'inningsPitched' of 76.1 means 76 AND 1/3."""
    raw = pd.DataFrame(
        {
            "playerId": [1, 2],
            "season": [2015, 2015],
            "age": [27, 27],
            "outs": [229, 200],
            "inningsPitched": ["76.1", "66.2"],
            "earnedRuns": [30, 30],
            "hits": [70, 70],
            "baseOnBalls": [20, 20],
            "gamesPitched": [12, 60],
            "gamesStarted": [12, 0],
            "wins": [5, 2],
            "strikeOuts": [80, 70],
            "saves": [0, 25],
        }
    )
    prepared = prepare_pitchers(raw)
    assert float(prepared["VOL"].iloc[0]) == pytest.approx(229 / 3), (
        f"IP must be outs/3 = {229 / 3:.3f}, got {float(prepared['VOL'].iloc[0])}. "
        f"Reading '76.1' as a float understates every rate denominator."
    )
    assert float(prepared["ERA"].iloc[0]) == pytest.approx(9 * 30 / (229 / 3))
    assert list(prepared["role"]) == ["starter", "reliever"], (
        f"Role split failed: {list(prepared['role'])}. A 12-start pitcher is a "
        f"starter; a 60-appearance 0-start pitcher is a reliever."
    )


def test_cumulative_decay_boundaries():
    table = pd.DataFrame(
        {
            "role": "hitter",
            "category": "OPS",
            "decile": -1,
            "variant": "inclusive",
            "age": list(range(MIN_AGE, MAX_AGE + 1)),
            "factor": 0.98,
            "n": 100,
            "league_factor": 1.0,
        }
    )
    assert cumulative_decay(table, 27, 0, "OPS", "hitter") == 1.0, (
        "Horizon 0 must be exactly 1.0 — no decay has happened yet."
    )
    assert cumulative_decay(table, 27, 2, "OPS", "hitter") == pytest.approx(0.98**2)

    with pytest.raises(AssertionError, match="below the fitted band"):
        cumulative_decay(table, 17, 5, "OPS", "hitter")

    assert cumulative_decay(table, MAX_AGE - 1, 5, "OPS", "hitter") == 0.0, (
        "Past MAX_AGE the player is declared finished, returning 0.0, rather "
        "than extrapolating a curve that was never fit there."
    )

    with pytest.raises(AssertionError, match="no cells"):
        cumulative_decay(table, 27, 2, "SB", "hitter")


def test_survival_factor_refuses_to_guess():
    table = pd.DataFrame(
        {
            "role": ["hitter"],
            "age_band": ["27-29"],
            "k": [1],
            "n": [500],
            "rate": [0.8],
            "base_floor": [500.0],
            "later_floor": [400.0],
        }
    )
    assert survival_factor(table, 27, 0, "hitter") == 1.0
    assert survival_factor(table, 27, 1, "hitter") == pytest.approx(0.8)
    # Any age inside the band resolves to the same cell -- that is the point of
    # banding, and it is explicit rather than a silent neighbour substitution.
    assert survival_factor(table, 29, 1, "hitter") == pytest.approx(0.8)
    # A band with no cell must refuse rather than borrow the neighbouring one.
    with pytest.raises(AssertionError, match="do not substitute"):
        survival_factor(table, 31, 1, "hitter")
    # Outside the banded range at all.
    with pytest.raises(AssertionError, match="outside the banded range"):
        survival_factor(table, 17, 1, "hitter")
