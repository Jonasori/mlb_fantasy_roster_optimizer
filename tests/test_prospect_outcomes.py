"""Offline guardrails for data_prep.prospect_outcomes.

No network, no mocking, no fixtures — every test builds the smallest synthetic
frame that would break if the logic broke.
"""

import numpy as np
import pandas as pd
import pytest

from data_prep.prospect_outcomes import (
    AGE_REL_LABELS,
    COHORT_SEASONS,
    MIN_CELL_N,
    MIN_GRADEABLE_SEASON_N,
    OUTCOME_WINDOW_YEARS,
    PERF_LABELS,
    STARTER_SLOTS,
    STAR_SLOTS,
    TIERS,
    _page_params,
    _rate_table,
    add_level_context,
    assign_tier,
    build_arrival_hazard,
    build_cohort,
    build_outcome_rates,
    bucket_age_rel,
    bucket_perf,
    grade_mlb_seasons,
    outcome_prior,
)

# Column order the unified season-line schema uses. Kept here rather than
# imported so a silent rename upstream fails a test instead of passing.
_STATE_COLS = [
    "player_id", "name", "position_type", "age", "G", "PA", "R", "HR", "RBI",
    "SB", "OPS", "IP", "W", "SV", "K", "ER", "HA", "BBA", "ERA", "WHIP",
    "season", "sport_id", "level", "player_type",
]


def _hitter_row(player_id: int, season: int, sport_id: int, age: float, pa: int,
                ops: float, name: str = "Test Player") -> dict:
    """One synthetic minor- or major-league hitting line."""
    return {
        "player_id": player_id, "name": name, "position_type": "Outfielder",
        "age": age, "G": pa // 4, "PA": float(pa), "R": pa * 0.13,
        "HR": pa * 0.03, "RBI": pa * 0.12, "SB": pa * 0.02, "OPS": ops,
        "IP": 0.0, "W": 0.0, "SV": 0.0, "K": 0.0, "ER": 0.0, "HA": 0.0,
        "BBA": 0.0, "ERA": 0.0, "WHIP": 0.0, "season": season,
        "sport_id": sport_id, "level": "AA" if sport_id == 12 else "MLB",
        "player_type": "hitter",
    }


def _pitcher_row(player_id: int, season: int, sport_id: int, age: float,
                 ip: float, era: float, whip: float, k: float = 0.0,
                 sv: float = 0.0, name: str = "Test Pitcher") -> dict:
    """One synthetic minor- or major-league pitching line."""
    return {
        "player_id": player_id, "name": name, "position_type": "Pitcher",
        "age": age, "G": int(ip // 5) + 1, "PA": 0.0, "R": 0.0, "HR": 0.0,
        "RBI": 0.0, "SB": 0.0, "OPS": 0.0, "IP": ip, "W": ip / 20.0, "SV": sv,
        "K": k if k else ip * 1.0, "ER": era * ip / 9.0, "HA": whip * ip * 0.75,
        "BBA": whip * ip * 0.25, "ERA": era, "WHIP": whip, "season": season,
        "sport_id": sport_id, "level": "AA" if sport_id == 12 else "MLB",
        "player_type": "pitcher",
    }


# ---------------------------------------------------------------------------
# Gradeable MLB season populations
# ---------------------------------------------------------------------------

# grade_mlb_seasons standardizes FV WITHIN each (season, player_type) over the
# rosterable rows only, so any MLB season a test feeds it needs a real
# population on both sides: at least MIN_GRADEABLE_SEASON_N rows, and genuine
# spread in all ten categories. Every pool below is built from np.linspace, so
# the suite is deterministic with no RNG at all.
_POOL_N = MIN_GRADEABLE_SEASON_N + 10


def _mlb_hitter_pool(season: int, n: int, first_id: int) -> list[dict]:
    """n rosterable MLB hitting lines, strictly ordered best to worst.

    Playing time falls from every-day to platoon, and the counting stats in
    `_hitter_row` are PA-derived, so row i beats row i + 1 in R, HR, RBI, SB
    AND OPS. FV rank is therefore exactly i + 1.
    """
    pa = np.linspace(700.0, 300.0, n)
    ops = np.linspace(0.950, 0.550, n)
    return [
        _hitter_row(first_id + i, season, 1, age=28.0, pa=int(pa[i]),
                    ops=float(ops[i]), name=f"PoolH{first_id + i}")
        for i in range(n)
    ]


def _mlb_pitcher_pool(season: int, n: int, first_id: int) -> list[dict]:
    """n rosterable MLB pitching lines with spread in all five pitching cats.

    Innings fall from ace-starter to closer workload, taking W (= IP / 20) and
    K down with them, while ERA and WHIP climb; the short-outing end of the
    pool carries the saves, so SV has real variance instead of being all zero.
    """
    ip = np.linspace(210.0, 55.0, n)
    era = np.linspace(2.40, 5.60, n)
    whip = np.linspace(0.92, 1.68, n)
    k = np.linspace(250.0, 45.0, n)
    sv = np.linspace(0.0, 40.0, n)
    return [
        _pitcher_row(first_id + i, season, 1, age=28.0, ip=float(ip[i]),
                     era=float(era[i]), whip=float(whip[i]), k=float(k[i]),
                     sv=float(sv[i]), name=f"PoolP{first_id + i}")
        for i in range(n)
    ]


def _mlb_season_pool(season: int, n_hitters: int = _POOL_N,
                     n_pitchers: int = _POOL_N) -> list[dict]:
    """Both sides of one gradeable MLB season, as background population.

    The ids are far away from the handful of ids the tests assert on, and are
    reused across seasons on purpose: a filler is the same player every year.
    """
    return (_mlb_hitter_pool(season, n_hitters, first_id=100_000)
            + _mlb_pitcher_pool(season, n_pitchers, first_id=200_000))


# ---------------------------------------------------------------------------
# The playerPool trap
# ---------------------------------------------------------------------------


def test_page_params_always_sends_player_pool_all():
    """Dropping playerPool silently returns ~158 of ~1100 players at a level."""
    params = _page_params(2010, 12, "hitting", 0)
    assert params["playerPool"] == "all", (
        f"_page_params must send playerPool=all; got {params}."
    )
    assert params["sportId"] == 12 and params["season"] == 2010


# ---------------------------------------------------------------------------
# Age relative to level
# ---------------------------------------------------------------------------


def test_age_rel_is_age_minus_level_season_mean():
    """age_rel arithmetic, and that it is computed per (season, level, type)."""
    rows = [
        _hitter_row(1, 2010, 12, age=20.0, pa=400, ops=0.800),
        _hitter_row(2, 2010, 12, age=24.0, pa=400, ops=0.700),
        _hitter_row(3, 2010, 12, age=25.0, pa=400, ops=0.700),
        _hitter_row(4, 2010, 12, age=27.0, pa=400, ops=0.700),
        # A different level-season: mean age 30, so its 30-year-old is average.
        _hitter_row(5, 2010, 11, age=30.0, pa=400, ops=0.700),
    ]
    out = add_level_context(pd.DataFrame(rows)[_STATE_COLS])
    aa = out.loc[out["sport_id"] == 12].set_index("player_id")
    # AA mean age = (20 + 24 + 25 + 27) / 4 = 24.0
    assert aa.loc[1, "level_mean_age"] == pytest.approx(24.0)
    assert aa.loc[1, "age_rel"] == pytest.approx(-4.0)
    assert aa.loc[4, "age_rel"] == pytest.approx(3.0)
    aaa = out.loc[out["sport_id"] == 11].set_index("player_id")
    assert aaa.loc[5, "age_rel"] == pytest.approx(0.0), (
        "age_rel must be relative to the player's OWN level-season, not a "
        "pooled mean across levels."
    )


def test_age_rel_buckets_are_left_closed_at_the_stated_edges():
    """Boundary behaviour of the age_rel buckets, which are right-open."""
    values = pd.Series([-3.0, -2.0, -1.0, -0.001, 0.0, 0.999, 1.0, 5.0])
    got = list(bucket_age_rel(values))
    assert got == [
        "<-2", "-2..-1", "-1..0", "-1..0", "0..1", "0..1", ">=1", ">=1",
    ], f"unexpected age_rel bucketing: {got}"
    assert set(got) <= set(AGE_REL_LABELS)


def test_perf_buckets_center_on_the_level_average_of_100():
    values = pd.Series([50.0, 89.9, 90.0, 100.0, 110.0, 124.9, 125.0, 200.0])
    got = list(bucket_perf(values))
    assert got == [
        "<90", "<90", "90-100", "100-110", "110-125", "110-125", "125+", "125+",
    ], f"unexpected perf bucketing: {got}"
    assert set(got) <= set(PERF_LABELS)
    assert pd.isna(bucket_perf(pd.Series([np.nan])).iloc[0])


def test_performance_index_is_relative_to_the_level_season():
    """A 130 OPS-index must mean 30% above that level's own aggregate OPS."""
    rows = [
        _hitter_row(1, 2010, 12, age=21.0, pa=500, ops=0.900),
        _hitter_row(2, 2010, 12, age=23.0, pa=500, ops=0.700),
        _hitter_row(3, 2010, 12, age=23.0, pa=500, ops=0.700),
        _hitter_row(4, 2010, 12, age=23.0, pa=500, ops=0.700),
    ]
    out = add_level_context(pd.DataFrame(rows)[_STATE_COLS]).set_index("player_id")
    # PA-weighted level OPS = (0.900 + 0.700*3) / 4 = 0.750
    assert out.loc[1, "OPS_index"] == pytest.approx(100.0 * 0.900 / 0.750)
    assert out.loc[1, "perf_index"] == pytest.approx(out.loc[1, "OPS_index"]), (
        "perf_index for a hitter is OPS_index."
    )
    assert out.loc[2, "OPS_index"] == pytest.approx(100.0 * 0.700 / 0.750)


def test_pitcher_perf_index_rewards_low_era_and_whip():
    """ERA/WHIP indices are reflected, so better than the level scores above 100."""
    rows = [
        _pitcher_row(1, 2010, 12, age=21.0, ip=100.0, era=2.00, whip=1.00, k=120),
        _pitcher_row(2, 2010, 12, age=23.0, ip=100.0, era=4.00, whip=1.40, k=80),
        _pitcher_row(3, 2010, 12, age=23.0, ip=100.0, era=4.00, whip=1.40, k=80),
    ]
    out = add_level_context(pd.DataFrame(rows)[_STATE_COLS]).set_index("player_id")
    assert out.loc[1, "perf_index"] > 100.0 < out.loc[2, "perf_index"] + 1e9
    assert out.loc[1, "perf_index"] > out.loc[2, "perf_index"], (
        "the better pitcher must have the higher perf_index"
    )
    assert out.loc[2, "ERA_index"] < 100.0 or out.loc[2, "ERA_index"] == pytest.approx(
        100.0 * (2.0 - 4.0 / out.loc[2, "pool_ERA"])
    )
    assert pd.isna(out.loc[1, "OPS_index"]), (
        "a pitcher must not carry a hitting index built from the zero fills"
    )


def test_pitchers_do_not_pollute_the_hitting_baseline():
    """A pitcher's handful of plate appearances is not a hitting observation."""
    rows = [
        _hitter_row(1, 2010, 12, age=22.0, pa=500, ops=0.750),
        _hitter_row(2, 2010, 12, age=23.0, pa=500, ops=0.750),
        dict(_hitter_row(9, 2010, 12, age=28.0, pa=300, ops=0.200),
             position_type="Pitcher"),
    ]
    out = add_level_context(pd.DataFrame(rows)[_STATE_COLS])
    assert 9 not in set(out["player_id"]), (
        "hitting rows belonging to pitchers must be dropped before the level "
        "baseline is computed"
    )
    assert out["level_mean_age"].iloc[0] == pytest.approx(22.5)


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


def test_assign_tier_boundaries():
    """Exact boundaries of the four fantasy-denominated tiers."""
    frame = pd.DataFrame(
        {
            "reached": [False, True, True, True, True, True, False],
            "starter": [0, 0, 1, 2, 5, 0, 0],
            "star": [0, 0, 0, 0, 0, 1, 0],
        }
    )
    got = list(assign_tier(frame["reached"], frame["starter"], frame["star"]))
    assert got == [
        "never",    # no MLB at all
        "fringe",   # reached, nothing more
        "fringe",   # ONE starter-grade season is not a dynasty regular
        "regular",  # two is the boundary
        "regular",
        "star",     # one star season outranks any number of starter seasons
        "never",
    ], f"tier boundaries moved: {got}"
    assert set(got) <= set(TIERS)


def test_star_beats_regular_even_with_one_mlb_season():
    got = assign_tier(pd.Series([True]), pd.Series([1]), pd.Series([1])).iloc[0]
    assert got == "star", f"star must take precedence, got {got}"


def test_grade_mlb_seasons_uses_our_league_slot_counts():
    """Grades come from FV rank against OUR slot counts, not a WAR threshold."""
    n_h = STARTER_SLOTS["hitter"] + 40
    n_p = STARTER_SLOTS["pitcher"] + 40
    rows = (_mlb_hitter_pool(2015, n_h, first_id=1000)
            + _mlb_pitcher_pool(2015, n_p, first_id=5000))
    # A part-timer: reached MLB, never rosterable, so never graded.
    rows.append(_hitter_row(1, 2015, 1, age=24.0, pa=40, ops=1.400, name="CupOfCoffee"))
    graded = grade_mlb_seasons(pd.DataFrame(rows)[_STATE_COLS]).set_index("player_id")
    hitters = graded.loc[graded["player_type"] == "hitter"]
    pitchers = graded.loc[graded["player_type"] == "pitcher"]

    assert not graded.loc[1, "rosterable"], "40 PA is not a rosterable season"
    assert pd.isna(graded.loc[1, "FV"]) and not graded.loc[1, "starter_grade"], (
        "an unrosterable season must not be graded, however good its rate stats"
    )
    # Counted per side: the two slot counts are different numbers and the ranks
    # are struck within (season, player_type), so a pooled count would pass on
    # the wrong total.
    assert int(hitters["starter_grade"].sum()) == STARTER_SLOTS["hitter"]
    assert int(hitters["star_grade"].sum()) == STAR_SLOTS["hitter"]
    assert int(pitchers["starter_grade"].sum()) == STARTER_SLOTS["pitcher"]
    assert int(pitchers["star_grade"].sum()) == STAR_SLOTS["pitcher"]
    assert graded.loc[1000, "star_grade"], "the best hitter must be star-grade"
    assert not graded.loc[1000 + n_h - 1, "starter_grade"], (
        "the worst of n > slots hitters must not be starter-grade"
    )


def test_grade_mlb_seasons_refuses_a_season_too_small_to_standardize():
    """A short season block means the fetch failed, not that baseball shrank.

    These 49 hitters have perfectly good variance, so `add_fantasy_value` would
    happily z-score them; ranking them against a population that size is the
    thing that is wrong, and only grade_mlb_seasons can see that.
    """
    rows = (_mlb_hitter_pool(2015, MIN_GRADEABLE_SEASON_N - 1, first_id=1000)
            + _mlb_pitcher_pool(2015, _POOL_N, first_id=5000))
    with pytest.raises(AssertionError, match="MIN_GRADEABLE_SEASON_N"):
        grade_mlb_seasons(pd.DataFrame(rows)[_STATE_COLS])


# ---------------------------------------------------------------------------
# Censoring
# ---------------------------------------------------------------------------


def _tiny_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A minor-league frame spanning the cohort edge, plus its MLB outcomes.

    Player 1: 2010 AA, reaches MLB in 2013 as a starter for two years.
    Player 2: 2010 AA, never reaches MLB.
    Player 3: 2010 AA, already an MLB player in 2008 — not a prospect.
    Player 4: 2021 AA (post-cohort, censored) — must be excluded.
    """
    milb = pd.DataFrame(
        [
            _hitter_row(1, 2010, 12, age=21.0, pa=500, ops=0.900, name="Riser"),
            _hitter_row(2, 2010, 12, age=25.0, pa=500, ops=0.650, name="Washout"),
            _hitter_row(3, 2010, 12, age=27.0, pa=500, ops=0.700, name="Demoted"),
            _hitter_row(4, 2021, 12, age=21.0, pa=500, ops=0.900, name="Censored"),
        ]
    )[_STATE_COLS]
    mlb_rows = [
        _hitter_row(3, 2008, 1, age=25.0, pa=500, ops=0.750, name="Demoted"),
        _hitter_row(3, 2012, 1, age=29.0, pa=500, ops=0.750, name="Demoted"),
        _hitter_row(4, 2023, 1, age=23.0, pa=600, ops=0.900, name="Censored"),
    ]
    for season in (2013, 2014):
        mlb_rows.append(
            _hitter_row(1, season, 1, age=24.0, pa=600, ops=1.000, name="Riser")
        )
    # Every MLB season named above has to be gradeable on its own, so each one
    # gets a full population on both sides. Riser's two seasons get more hitters
    # than we have starting slots, so slot-based grading actually cuts.
    for season in (2008, 2012, 2023):
        mlb_rows += _mlb_season_pool(season)
    for season in (2013, 2014):
        mlb_rows += _mlb_season_pool(season, n_hitters=STARTER_SLOTS["hitter"] + 20)
    return milb, pd.DataFrame(mlb_rows)[_STATE_COLS]


def test_censored_cohorts_are_excluded():
    """A post-2018 minor-league season cannot be observed for 8 years yet."""
    milb, mlb = _tiny_universe()
    cohort = build_cohort(milb, mlb)
    assert set(cohort["season"]) <= set(COHORT_SEASONS), (
        f"cohort contains seasons outside {COHORT_SEASONS}: "
        f"{sorted(set(cohort['season']))}. A censored cohort silently deflates "
        f"every success rate."
    )
    assert 4 not in set(cohort["player_id"]), (
        "the 2021 observation must be dropped even though its outcome looks good"
    )


def test_players_already_in_mlb_are_not_prospects():
    milb, mlb = _tiny_universe()
    cohort = build_cohort(milb, mlb)
    assert 3 not in set(cohort["player_id"]), (
        "a minor leaguer who had already debuted is a demoted major leaguer; "
        "including him inflates every arrival rate"
    )


def test_outcomes_respect_the_eight_year_window():
    """An MLB season outside [milb_season, +8] must not count."""
    milb = pd.DataFrame(
        [_hitter_row(1, 2010, 12, age=21.0, pa=500, ops=0.900, name="Slow")]
    )[_STATE_COLS]
    late = 2010 + OUTCOME_WINDOW_YEARS + 1
    mlb = pd.DataFrame(
        [_hitter_row(1, late, 1, age=32.0, pa=600, ops=0.900, name="Slow")]
        + _mlb_season_pool(late)
    )[_STATE_COLS]
    cohort = build_cohort(milb, mlb)
    assert len(cohort) == 1
    assert not bool(cohort["reached"].iloc[0]), (
        f"an arrival in {late}, more than {OUTCOME_WINDOW_YEARS} years after "
        f"the 2010 observation, is outside the window and must read as never"
    )
    assert cohort["tier"].iloc[0] == "never"
    assert cohort["arrival"].iloc[0] == f"never_within_{OUTCOME_WINDOW_YEARS}"


def test_arrival_year_is_measured_from_the_observation_season():
    milb = pd.DataFrame(
        [_hitter_row(1, 2010, 12, age=21.0, pa=500, ops=0.900, name="Riser")]
    )[_STATE_COLS]
    mlb = pd.DataFrame(
        [
            _hitter_row(1, 2013, 1, age=24.0, pa=600, ops=0.800, name="Riser"),
            _hitter_row(1, 2014, 1, age=25.0, pa=600, ops=0.800, name="Riser"),
        ]
        + _mlb_season_pool(2013)
        + _mlb_season_pool(2014)
    )[_STATE_COLS]
    cohort = build_cohort(milb, mlb)
    assert int(cohort["years_to_mlb"].iloc[0]) == 3
    assert cohort["arrival"].iloc[0] == "3"
    assert cohort["age_at_first_mlb"].iloc[0] == pytest.approx(24.0), (
        "age_at_first_mlb must come from the DEBUT season's row, not the last"
    )


# ---------------------------------------------------------------------------
# Rate tables
# ---------------------------------------------------------------------------


def test_rate_table_is_long_with_n_and_sums_to_one_per_cell():
    frame = pd.DataFrame(
        {
            "player_type": ["hitter"] * 5,
            "tier": ["never", "never", "never", "fringe", "star"],
        }
    )
    long = _rate_table(frame, ["player_type"], "tier", list(TIERS), "test")
    assert set(long["tier"]) == set(TIERS), (
        "every tier must appear in every cell, including the zero ones, or a "
        "prior read off the table sums to less than 1"
    )
    assert long["p"].sum() == pytest.approx(1.0)
    assert set(long["cell_n"]) == {5}
    assert long.loc[long["tier"] == "never", "n"].iloc[0] == 3
    assert bool(long["sparse"].iloc[0]) is True, (
        f"a 5-row cell is below MIN_CELL_N={MIN_CELL_N} and must be flagged"
    )


def test_rate_tables_carry_both_conditionings_and_a_denominator():
    milb, mlb = _tiny_universe()
    cohort = build_cohort(milb, mlb)
    rates = build_outcome_rates(milb, mlb, cohort=cohort)
    assert set(rates["conditioning"]) == {"full", "age_level"}
    marginal = rates.loc[rates["conditioning"] == "age_level"]
    assert set(marginal["age_rel_bucket"]) == {"ALL"}
    assert (rates["cell_n"] > 0).all(), "every emitted cell needs an n"
    for _, cell in rates.groupby(
        ["conditioning", "player_type", "age", "sport_id", "age_rel_bucket",
         "perf_bucket"]
    ):
        assert cell["p"].sum() == pytest.approx(1.0)


def test_arrival_hazard_sums_to_one_including_never():
    milb, mlb = _tiny_universe()
    cohort = build_cohort(milb, mlb)
    hazard = build_arrival_hazard(milb, mlb, cohort=cohort)
    assert f"never_within_{OUTCOME_WINDOW_YEARS}" in set(hazard["arrival"]), (
        "the hazard table must carry the never-arrives mass, or the ETA "
        "distribution integrates to the arrival rate instead of to 1"
    )
    for _, cell in hazard.groupby(["player_type", "age", "sport_id"]):
        assert cell["p"].sum() == pytest.approx(1.0)
        assert len(cell) == OUTCOME_WINDOW_YEARS + 2


# ---------------------------------------------------------------------------
# outcome_prior
# ---------------------------------------------------------------------------


def _write_rates(tmp_path, cells: list[dict], monkeypatch) -> None:
    """Point the module's rate-table path at a synthetic file."""
    from data_prep import prospect_outcomes as po

    rows = []
    for cell in cells:
        for tier, p, n in zip(TIERS, cell["p"], cell["n"]):
            rows.append({**{k: v for k, v in cell.items() if k not in ("p", "n")},
                         "tier": tier, "p": p, "n": n,
                         "cell_n": sum(cell["n"]),
                         "sparse": sum(cell["n"]) < MIN_CELL_N,
                         "conditioning": "full"})
    path = tmp_path / "outcome_rates.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    monkeypatch.setattr(po, "OUTCOME_RATES_PATH", path)
    po._load_outcome_rates.cache_clear()


_POPULATED = {
    "player_type": "hitter", "age": 21, "sport_id": 12, "level": "AA",
    "age_rel_bucket": "-2..-1", "perf_bucket": "125+",
    "p": [0.30, 0.30, 0.25, 0.15], "n": [30, 30, 25, 15],
}


def test_outcome_prior_sums_to_one_including_never_reached(tmp_path, monkeypatch):
    _write_rates(tmp_path, [_POPULATED], monkeypatch)
    prior = outcome_prior(21, 12, -1.5, 130.0, "hitter")
    assert set(prior) == set(TIERS), (
        f"the prior must span every tier, got {sorted(prior)}"
    )
    assert sum(prior.values()) == pytest.approx(1.0), (
        "a prior summing to less than 1 silently discounts every player by the "
        "missing mass"
    )
    assert prior["never"] == pytest.approx(0.30)
    assert prior["star"] == pytest.approx(0.15)


def test_outcome_prior_asserts_on_an_unpopulated_cell(tmp_path, monkeypatch):
    _write_rates(tmp_path, [_POPULATED], monkeypatch)
    with pytest.raises(AssertionError, match="never observed"):
        outcome_prior(19, 11, 3.0, 60.0, "hitter")


def test_outcome_prior_refuses_a_cell_below_the_sparsity_floor(tmp_path, monkeypatch):
    thin = {**_POPULATED, "n": [3, 3, 2, 1], "p": [0.3333, 0.3333, 0.2222, 0.1112]}
    _write_rates(tmp_path, [thin], monkeypatch)
    with pytest.raises(AssertionError, match=f"below MIN_CELL_N={MIN_CELL_N}"):
        outcome_prior(21, 12, -1.5, 130.0, "hitter")


def test_outcome_prior_rejects_pooled_or_nonsense_arguments(tmp_path, monkeypatch):
    _write_rates(tmp_path, [_POPULATED], monkeypatch)
    with pytest.raises(AssertionError, match="never pooled"):
        outcome_prior(21, 12, -1.5, 130.0, "both")
    with pytest.raises(AssertionError, match="minor-league sportId"):
        outcome_prior(21, 1, -1.5, 130.0, "hitter")
