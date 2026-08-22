"""
Empirical P(career outcome | observable minor-league state), in OUR categories.

WHY THIS EXISTS
---------------
Published prospect research is denominated in WAR. WAR is defense-inclusive, so
it is the wrong unit for a 10-category roto league: a glove-first shortstop who
is a 3-WAR player can be a zero in R/HR/RBI/SB/OPS. Worse, the published FV-grade
base rates cannot even be pooled with one another — the same author reports 21%
and 10.4% bust for the same players in two papers, and the entire gap is
definitional. So we build our own rates, in our own categories, with our own
tier definitions stated out loud.

THE OUTCOME TIERS (ours, and the whole point of the module)
-----------------------------------------------------------
For every MLB season we compute FV = the sum of 5 category z-scores, using the
repo's own scorer (`optimizer.player_scoring.add_fantasy_value` — not
reimplemented here). z-scores are taken WITHIN (mlb_season, player_type) over
the ROSTERABLE pool only (hitters PA >= 250; pitchers IP >= 50 or SV >= 5),
because standardizing over all ~1400 players who touched a field would make
every full-time regular look like a star.

A season is then graded by its FV rank inside its (season, player_type) pool,
against OUR league's slot counts (7 teams x 11 hitting slots = 77 starting
hitters; 7 teams x 7 pitching slots = 49 starting pitchers):

    starter-grade : FV rank <= league starting slots for that type
    star-grade    : FV rank <= 25% of those slots (a genuine early-round asset)

and a player's career gets exactly one tier:

    star     : >= 1 star-grade season
    regular  : >= 2 starter-grade seasons  <-- the tier that matters most to us
    fringe   : reached MLB, but neither of the above
    never    : no MLB appearance at all

"regular" deliberately demands TWO starter-grade seasons. In a dynasty league a
single useful year followed by nothing is a fringe outcome, not a regular one.

THE 8-YEAR WINDOW (a correctness fix, not a shortcut)
-----------------------------------------------------
Every outcome — arrival, tier, career totals — is measured over the MLB seasons
in [milb_season, milb_season + OUTCOME_WINDOW_YEARS]. Without a fixed window a
2005 observation gets 21 years for its outcome to materialize while a 2018
observation gets 8, so the old cohorts look systematically better and the rate
is a blend of two different questions. A fixed window makes every cohort
identically observed, makes "never" well defined, and happens to be exactly the
conditioning a dynasty valuation model wants: roster time is the cost being paid.

Cohorts are minor-league seasons 2005-2018 only. Post-2018 cohorts are censored
against an 8-year window and would silently deflate every success rate.

JOINS ARE ON player.id (MLBAM). Never on name: a name join previously spliced
Max Muncy (571970, 35yo) into Max Muncy (691777, 23yo) and produced a fictional
recommendation.
"""

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from optimizer.config import HITTING_SLOTS, NUM_OPPONENTS, PITCHING_SLOTS
from optimizer.player_scoring import add_fantasy_value

from .names import strip_diacritics
from .raw_io import DATA_DIR, REPO_ROOT
from .statsapi_stats import parse_rate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATS_URL = "https://statsapi.mlb.com/api/v1/stats"

# MLB StatsAPI sport ids. NOTE: the `MLB-StatsAPI` package's `statsapi.get()`
# SILENTLY DROPS the sportId parameter and returns MLB rows with sport.id == 1,
# which is why this module talks to the endpoint with `requests` directly.
LEVELS: dict[int, str] = {
    1: "MLB", 11: "AAA", 12: "AA", 13: "A+", 14: "A", 15: "A-/RkAdv", 16: "R",
}

# Default minor-league levels. sportId 15 (short-season A / Rookie Advanced)
# existed through 2020 and returns ~450-550 hitters a season; pass it in
# explicitly to include it. It is left out of the default so the fetched set
# matches the documented level list the tables were built from.
DEFAULT_SPORT_IDS: tuple[int, ...] = (11, 12, 13, 14, 16)

COHORT_SEASONS = range(2005, 2019)
OUTCOME_WINDOW_YEARS = 8
# Back to 1995 so a 2005 minor leaguer who had ALREADY debuted (and is
# therefore not a prospect) can be identified and excluded.
MLB_SEASONS = range(1995, 2027)

# Page size for the stats endpoint; the largest level-season is ~1900 rows.
_PAGE_LIMIT = 2000
_REQUEST_PAUSE_S = 0.15

# Playing-time gate. Used for BOTH the level-season normalization pool and
# cohort inclusion, so "the population you are compared against" and "the
# population you belong to" are the same set.
MIN_POOL_PA = 100
MIN_POOL_IP = 20.0

# Rosterable MLB pool: who FV is standardized over.
MIN_MLB_PA = 250
MIN_MLB_IP = 50.0
MIN_MLB_SV = 5

# Smallest rosterable (season, player_type) block that can be standardized.
# Every real MLB season 1995-2026 clears this by an order of magnitude, so a
# block below it means the fetch came back partial, not that baseball shrank.
MIN_GRADEABLE_SEASON_N = 50

# 2020 was 60 games. Only 16 hitters cleared MIN_MLB_PA, so it cannot be graded
# against full-season slot counts: FV is standardized within a season and the
# starter/star cutoffs are absolute rosterable ranks. Grading it would hand out
# star seasons to anyone who stayed healthy for two months, and withhold them
# from everyone else. Excluded outright, matching data_prep.aging.
UNGRADEABLE_SEASONS: frozenset[int] = frozenset({2020})

NUM_TEAMS = NUM_OPPONENTS + 1
STARTER_SLOTS: dict[str, int] = {
    "hitter": NUM_TEAMS * sum(HITTING_SLOTS.values()),
    "pitcher": NUM_TEAMS * sum(PITCHING_SLOTS.values()),
}
STAR_SLOT_FRACTION = 0.25
STAR_SLOTS: dict[str, int] = {
    k: int(round(v * STAR_SLOT_FRACTION)) for k, v in STARTER_SLOTS.items()
}

TIERS: tuple[str, ...] = ("never", "fringe", "regular", "star")

# Ages worth tabulating. Outside this the cells are retired-organizational-guy
# noise; the rows stay in the cohort parquet, they just leave the rate tables.
PRIOR_AGE_MIN = 17
PRIOR_AGE_MAX = 26

# Below this an estimated cell is flagged, never smoothed and never hidden.
MIN_CELL_N = 20

# Bucket edges. Negative age_rel means young for the level, which is the single
# most cited stats-only prospect indicator, so it gets its own first-class column.
AGE_REL_EDGES = (-np.inf, -2.0, -1.0, 0.0, 1.0, np.inf)
AGE_REL_LABELS = ("<-2", "-2..-1", "-1..0", "0..1", ">=1")
PERF_EDGES = (-np.inf, 90.0, 100.0, 110.0, 125.0, np.inf)
PERF_LABELS = ("<90", "90-100", "100-110", "110-125", "125+")

# The 10 scoring categories, split by side.
HIT_CATS: tuple[str, ...] = ("R", "HR", "RBI", "SB", "OPS")
PITCH_CATS: tuple[str, ...] = ("W", "SV", "K", "ERA", "WHIP")

SEASON_LINES_DIR = DATA_DIR / "raw" / "season_lines"
PRIORS_DIR = DATA_DIR / "priors"
COHORT_PATH = PRIORS_DIR / "milb_cohort.parquet"
OUTCOME_RATES_PATH = PRIORS_DIR / "outcome_rates.parquet"
ARRIVAL_HAZARD_PATH = PRIORS_DIR / "arrival_hazard.parquet"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _parse_splits(
    splits: list[dict], group: str, season: int, sport_id: int
) -> pd.DataFrame:
    """Flatten one level-season's splits into the unified season-line schema.

    Args:
        splits: `stats[0]["splits"]` entries from the stats endpoint.
        group: "hitting" or "pitching".
        season: Season the splits belong to (the payload's is a string).
        sport_id: Level the splits were requested for.

    Returns:
        One row per player. Columns: player_id, name, season, sport_id, level,
        player_type, position_type, age, G, PA, R, HR, RBI, SB, OPS, IP, W, SV,
        K, ERA, WHIP. The unused side is filled with 0.0 so the frame satisfies
        `add_fantasy_value`'s column contract as-is.

    Note:
        ERA and WHIP are DERIVED from earnedRuns / hits / walks and `outs`,
        never read from the string fields. `inningsPitched` "76.1" means 76 AND
        ONE THIRD, and the rate strings use '.---' / '-.--' as missing sentinels.
    """
    assert group in ("hitting", "pitching"), (
        f"_parse_splits: group must be 'hitting' or 'pitching', got {group!r}. "
        f"Those are the only two the stats endpoint exposes."
    )
    rows = []
    n_unknown = 0
    for split in splits:
        # player.id 0 is the API's unknown-player sentinel (it also carries no
        # fullName). It shows up in a handful of DSL/complex box scores. It is
        # not a player, so it cannot have an outcome; dropping it is the only
        # correct read. Counted, not swallowed.
        if int(split["player"]["id"]) == 0:
            n_unknown += 1
            continue
        stat = split["stat"]
        # `fullName` is genuinely absent for a small number of real players:
        # 2012 Triple-A pitching returns one split of 1202 whose player object
        # is only {'id', 'link'} (id 529900). The id is the join key and is
        # always present; the name is display-only. Recording the absence is
        # therefore correct, and is NOT a fallback — nothing is substituted, and
        # no downstream join depends on it. Joining on name instead of id is
        # what previously spliced two different Max Muncys together.
        rows.append(
            {
                "player_id": int(split["player"]["id"]),
                "name": split["player"].get(
                    "fullName", f"mlbam-{int(split['player']['id'])}"
                ),
                "position_type": (split.get("position") or {}).get("type"),
                "age": stat.get("age"),
                "G": stat.get("gamesPlayed"),
                "PA": stat.get("plateAppearances"),
                "R": stat.get("runs"),
                "HR": stat.get("homeRuns"),
                "RBI": stat.get("rbi"),
                "SB": stat.get("stolenBases"),
                "ops": stat.get("ops"),
                "outs": stat.get("outs"),
                "W": stat.get("wins"),
                "SV": stat.get("saves"),
                "K": stat.get("strikeOuts"),
                "ER": stat.get("earnedRuns"),
                "HA": stat.get("hits"),
                "BBA": stat.get("baseOnBalls"),
            }
        )
    assert rows, (
        f"_parse_splits: no splits for season={season} sportId={sport_id} "
        f"group={group}. An empty response here almost always means "
        f"playerPool=all was dropped from the query — check _page_params."
    )
    if n_unknown:
        print(
            f"  {season} sportId={sport_id} {group}: dropped {n_unknown} "
            f"unknown-player (id=0) split(s)"
        )
    frame = pd.DataFrame(rows)
    frame["season"] = int(season)
    frame["sport_id"] = int(sport_id)
    frame["level"] = LEVELS[sport_id]
    frame["player_type"] = "hitter" if group == "hitting" else "pitcher"

    numeric = ["age", "G", "PA", "R", "HR", "RBI", "SB", "outs", "W", "SV", "K",
               "ER", "HA", "BBA"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    frame["OPS"] = parse_rate(frame["ops"])
    frame = frame.drop(columns=["ops"])

    if group == "hitting":
        # Hitters never pitch: zeros keep add_fantasy_value's pitching terms inert.
        frame["IP"] = 0.0
        for col in ("W", "SV", "K", "ER", "HA", "BBA"):
            frame[col] = 0.0
        frame["ERA"] = 0.0
        frame["WHIP"] = 0.0
    else:
        frame["IP"] = frame["outs"] / 3.0
        has_ip = frame["IP"] > 0
        frame["ERA"] = np.where(has_ip, 9.0 * frame["ER"] / frame["IP"].where(has_ip), np.nan)
        frame["WHIP"] = np.where(
            has_ip, (frame["HA"] + frame["BBA"]) / frame["IP"].where(has_ip), np.nan
        )
        # Pitchers' own plate appearances are not a fantasy category for us.
        frame["PA"] = 0.0
        for col in ("R", "HR", "RBI", "SB"):
            frame[col] = 0.0
        frame["OPS"] = 0.0

    frame = frame.drop(columns=["outs"])

    assert not frame["player_id"].duplicated().any(), (
        f"_parse_splits: duplicate player_ids for season={season} "
        f"sportId={sport_id} group={group}. The endpoint is supposed to "
        f"aggregate a traded player across teams; a duplicate means the "
        f"response is split by team and the rows must be summed first."
    )
    return frame


def _page_params(season: int, sport_id: int, group: str, offset: int) -> dict:
    """Query params for one page of a level-season leaderboard.

    Pulled out so the playerPool trap is covered by an offline test: without
    `playerPool=all` the endpoint silently returns ~158 qualified players
    instead of the ~1100 who actually played the level.
    """
    return {
        "stats": "season",
        "group": group,
        "season": season,
        "sportId": sport_id,
        # MANDATORY. Omitting this drops ~85% of the level.
        "playerPool": "all",
        "limit": _PAGE_LIMIT,
        "offset": offset,
    }


def fetch_season_lines(season: int, sport_id: int, group: str) -> pd.DataFrame:
    """Fetch one (season, level, group) leaderboard, paginating on totalSplits.

    Returns:
        The unified season-line schema from `_parse_splits`.
    """
    splits: list[dict] = []
    offset = 0
    total = None
    while True:
        payload = requests.get(
            STATS_URL, params=_page_params(season, sport_id, group, offset), timeout=60
        ).json()
        blocks = payload["stats"]
        assert blocks, (
            f"fetch_season_lines: empty `stats` array for season={season} "
            f"sportId={sport_id} group={group}. That level-season may not "
            f"exist; check LEVELS and the season range."
        )
        block = blocks[0]
        total = block["totalSplits"]
        page = block["splits"]
        splits.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break
        time.sleep(_REQUEST_PAUSE_S)

    assert len(splits) == total, (
        f"fetch_season_lines: got {len(splits)} splits but totalSplits={total} "
        f"for season={season} sportId={sport_id} group={group}. Pagination "
        f"broke — check the offset loop before trusting the cache."
    )
    return _parse_splits(splits, group, season, sport_id)


def _cache_path(season: int, sport_id: int, group: str) -> Path:
    """Cache location for one (season, level, group) fetch.

    One file per fetch unit — the finest granularity a resumed run can skip.
    """
    return SEASON_LINES_DIR / f"{season}_{sport_id}_{group}.parquet"


def fetch_season_lines_cached(season: int, sport_id: int, group: str) -> pd.DataFrame:
    """Read one (season, level, group) from disk, fetching only if absent.

    This is the resume point: a run killed halfway costs nothing to restart.
    """
    path = _cache_path(season, sport_id, group)
    if path.exists():
        return pd.read_parquet(path)
    frame = fetch_season_lines(season, sport_id, group)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def _fetch_many(seasons: range, sport_ids: tuple[int, ...], label: str) -> pd.DataFrame:
    """Fetch every (season, level, group) in the grid, caching each to parquet."""
    units = [
        (season, sport_id, group)
        for season in seasons
        for sport_id in sport_ids
        for group in ("hitting", "pitching")
    ]
    cached = sum(1 for u in units if _cache_path(*u).exists())
    print(
        f"=== {label}: {len(units)} level-seasons "
        f"({cached} already on disk, {len(units) - cached} to fetch) ==="
    )
    frames = [
        fetch_season_lines_cached(*unit)
        for unit in tqdm(units, desc=f"{label} level-seasons")
    ]
    out = pd.concat(frames, ignore_index=True)
    print(
        f"=== {label}: {len(out):,} player-season-level rows, "
        f"{out['player_id'].nunique():,} distinct players ==="
    )
    return out


def fetch_milb_history(
    seasons: range = COHORT_SEASONS, sport_ids: tuple[int, ...] = DEFAULT_SPORT_IDS
) -> pd.DataFrame:
    """Every minor-league season line at the given levels, cached per fetch unit.

    Returns:
        The unified season-line schema, one row per (player, season, level,
        player_type). Hitting rows whose position type is "Pitcher" are kept
        here and dropped later in `add_level_context` — a pitcher's 4 plate
        appearances must not pollute the level's hitting baseline.
    """
    assert 1 not in sport_ids, (
        f"fetch_milb_history: sport_ids {sport_ids} contains 1 (MLB). MLB "
        f"seasons are the OUTCOME side — fetch them with fetch_mlb_outcomes."
    )
    return _fetch_many(seasons, sport_ids, "MiLB history")


def fetch_mlb_outcomes(seasons: range = MLB_SEASONS) -> pd.DataFrame:
    """Every MLB season line, cached per fetch unit.

    The range must start well before the first cohort season so that a minor
    leaguer who had ALREADY debuted can be recognised and excluded — he is a
    demoted major leaguer, not a prospect, and including him inflates every
    arrival rate.
    """
    return _fetch_many(seasons, (1,), "MLB outcomes")


# ---------------------------------------------------------------------------
# Observable state
# ---------------------------------------------------------------------------


def add_level_context(milb: pd.DataFrame) -> pd.DataFrame:
    """Normalize each minor-league line against its own level-season.

    Drops hitting rows belonging to pitchers (a pitcher's handful of plate
    appearances is not a hitting prospect observation, and it drags the level's
    hitting baseline down), then, within each (season, sport_id, player_type)
    pool of players clearing MIN_POOL_PA / MIN_POOL_IP, computes the level's
    aggregate rates and expresses every player against them.

    Requires columns: season, sport_id, player_type, position_type, age, PA, IP,
        R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP.
    Adds columns: in_pool, level_mean_age, age_rel, age_rel_bucket,
        R_index, HR_index, RBI_index, SB_index, OPS_index,
        W_index, SV_index, K_index, ERA_index, WHIP_index,
        perf_index, perf_bucket, name_ascii.

    Note:
        `perf_index` is OPS_index for hitters and the mean of
        {ERA_index, WHIP_index, K_index} for pitchers. Wins and saves are
        recorded as indices for completeness but excluded from the composite:
        in the minors they are role and team artifacts, not skill. All ten
        category indices are stored so a consumer can build a different
        composite without refetching.
    """
    milb = milb.copy()
    n_before = len(milb)
    is_pitcher_bat = (milb["player_type"] == "hitter") & (
        milb["position_type"] == "Pitcher"
    )
    milb = milb.loc[~is_pitcher_bat].copy()
    print(
        f"add_level_context: dropped {n_before - len(milb):,} pitcher-batting "
        f"rows of {n_before:,}"
    )

    milb["in_pool"] = np.where(
        milb["player_type"] == "hitter",
        milb["PA"] >= MIN_POOL_PA,
        milb["IP"] >= MIN_POOL_IP,
    )
    keys = ["season", "sport_id", "player_type"]
    pool = milb.loc[milb["in_pool"]]
    assert len(pool) > 0, (
        f"add_level_context: no rows clear MIN_POOL_PA={MIN_POOL_PA} / "
        f"MIN_POOL_IP={MIN_POOL_IP}. Check that PA and IP survived parsing."
    )

    # Level-season baselines, volume-weighted (AGENTS.md: ratio stats are
    # weighted averages, never means of per-player rates).
    weighted = pool.assign(
        _ops_w=pool["OPS"] * pool["PA"],
        _era_w=pool["ERA"] * pool["IP"],
        _whip_w=pool["WHIP"] * pool["IP"],
    )
    agg = weighted.groupby(keys).agg(
        level_mean_age=("age", "mean"),
        pool_PA=("PA", "sum"), pool_IP=("IP", "sum"),
        pool_R=("R", "sum"), pool_HR=("HR", "sum"), pool_RBI=("RBI", "sum"),
        pool_SB=("SB", "sum"), pool_W=("W", "sum"), pool_SV=("SV", "sum"),
        pool_K=("K", "sum"),
        _ops_w=("_ops_w", "sum"), _era_w=("_era_w", "sum"), _whip_w=("_whip_w", "sum"),
    )
    hitting_pool = agg.index.get_level_values("player_type") == "hitter"
    agg["pool_OPS"] = np.where(hitting_pool, agg["_ops_w"] / agg["pool_PA"], np.nan)
    agg["pool_ERA"] = np.where(hitting_pool, np.nan, agg["_era_w"] / agg["pool_IP"])
    agg["pool_WHIP"] = np.where(hitting_pool, np.nan, agg["_whip_w"] / agg["pool_IP"])
    agg = agg.drop(columns=["_ops_w", "_era_w", "_whip_w"]).reset_index()

    milb = milb.merge(agg, on=keys, how="left", validate="many_to_one")
    assert milb["level_mean_age"].notna().all(), (
        "add_level_context: some (season, sport_id, player_type) cells have no "
        "pooled players to average an age over. Every level-season in the "
        "fetch must have at least one player clearing the playing-time gate; "
        "check the fetch range and MIN_POOL_PA / MIN_POOL_IP."
    )
    milb["age_rel"] = milb["age"] - milb["level_mean_age"]
    milb["age_rel_bucket"] = bucket_age_rel(milb["age_rel"])

    hit = milb["player_type"] == "hitter"
    pa = milb["PA"].where(milb["PA"] > 0)
    ip = milb["IP"].where(milb["IP"] > 0)
    # index = 100 * (player rate) / (level-season rate); level average == 100.
    for cat in ("R", "HR", "RBI", "SB"):
        milb[f"{cat}_index"] = (
            100.0 * (milb[cat] / pa) / (milb[f"pool_{cat}"] / milb["pool_PA"])
        )
    milb["OPS_index"] = 100.0 * milb["OPS"] / milb["pool_OPS"]
    for cat in ("W", "SV", "K"):
        milb[f"{cat}_index"] = (
            100.0 * (milb[cat] / ip) / (milb[f"pool_{cat}"] / milb["pool_IP"])
        )
    # ERA and WHIP are negated categories, so the index is REFLECTED rather than
    # inverted: 100*(2 - player/level). Level average maps to 100, half the
    # level's ERA to 150, double to 0. A plain level/player ratio would divide
    # by a legitimate 0.00 ERA and produce infinity.
    milb["ERA_index"] = 100.0 * (2.0 - milb["ERA"] / milb["pool_ERA"])
    milb["WHIP_index"] = 100.0 * (2.0 - milb["WHIP"] / milb["pool_WHIP"])
    # A hitter has no pitching index and a pitcher has no hitting index; the
    # zero fills from parsing would otherwise read as real, terrible rates.
    for cat in PITCH_CATS:
        milb.loc[hit, f"{cat}_index"] = np.nan
    for cat in HIT_CATS:
        milb.loc[~hit, f"{cat}_index"] = np.nan

    milb["perf_index"] = np.where(
        hit,
        milb["OPS_index"],
        milb[["ERA_index", "WHIP_index", "K_index"]].mean(axis=1),
    )
    milb["perf_bucket"] = bucket_perf(milb["perf_index"])
    milb["name_ascii"] = milb["name"].map(strip_diacritics)
    print(
        f"add_level_context: {int(milb['in_pool'].sum()):,} of {len(milb):,} rows "
        f"clear the playing-time gate; median age_rel "
        f"{milb.loc[milb['in_pool'], 'age_rel'].median():+.2f}"
    )
    return milb


def bucket_age_rel(age_rel: pd.Series) -> pd.Series:
    """Bucket age-relative-to-level. Negative = young for the level = good."""
    return pd.cut(
        age_rel, bins=list(AGE_REL_EDGES), labels=list(AGE_REL_LABELS), right=False
    ).astype("string")


def bucket_perf(perf_index: pd.Series) -> pd.Series:
    """Bucket the level-normalized performance index (100 = level average)."""
    return pd.cut(
        perf_index, bins=list(PERF_EDGES), labels=list(PERF_LABELS), right=False
    ).astype("string")


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def grade_mlb_seasons(mlb: pd.DataFrame) -> pd.DataFrame:
    """Score every MLB season in fantasy terms and grade it against our league.

    FV is computed by `optimizer.player_scoring.add_fantasy_value` on the
    ROSTERABLE subset of each (season) — hitters PA >= MIN_MLB_PA, pitchers
    IP >= MIN_MLB_IP or SV >= MIN_MLB_SV. That subset is the population the
    z-scores are standardized over; everyone else keeps FV = NaN and both
    grades False, which is the correct statement that they were never a
    rosterable fantasy asset that year.

    Requires columns: season, player_id, player_type, PA, IP, and the 10
        category columns.
    Adds columns: rosterable, FV, fv_rank, starter_grade, star_grade.

    Note:
        Every season must supply at least MIN_GRADEABLE_SEASON_N rosterable
        players of EACH type, otherwise the within-season standardization is
        meaningless and this crashes rather than grading it.
    """
    mlb = mlb.copy()
    mlb["rosterable"] = np.where(
        mlb["player_type"] == "hitter",
        mlb["PA"] >= MIN_MLB_PA,
        (mlb["IP"] >= MIN_MLB_IP) | (mlb["SV"] >= MIN_MLB_SV),
    )
    mlb["FV"] = np.nan
    print(
        f"grade_mlb_seasons: {int(mlb['rosterable'].sum()):,} rosterable "
        f"season-lines of {len(mlb):,}; standardizing FV within each season"
    )
    ungradeable = mlb["season"].isin(UNGRADEABLE_SEASONS)
    if ungradeable.any():
        print(
            f"grade_mlb_seasons: excluding {sorted(UNGRADEABLE_SEASONS)} "
            f"({int((ungradeable & mlb['rosterable']).sum()):,} rosterable "
            f"season-lines). A 60-game season cannot be ranked against "
            f"full-season slot counts; those player-seasons keep FV = NaN and "
            f"both grades False."
        )
    mlb.loc[ungradeable, "rosterable"] = False

    scored = []
    for season, block in tqdm(
        mlb.loc[mlb["rosterable"]].groupby("season"), desc="FV by MLB season"
    ):
        for player_type in ("hitter", "pitcher"):
            n_type = int((block["player_type"] == player_type).sum())
            assert n_type >= MIN_GRADEABLE_SEASON_N, (
                f"grade_mlb_seasons: season {season} has only {n_type} "
                f"rosterable {player_type}s, below "
                f"MIN_GRADEABLE_SEASON_N={MIN_GRADEABLE_SEASON_N}. FV is "
                f"standardized WITHIN the season, so a block this small cannot "
                f"be ranked against its own population. Every MLB season "
                f"{MLB_SEASONS.start}-{MLB_SEASONS.stop - 1} has hundreds of "
                f"rosterable players, so one of two things is true. Either "
                f"{season} was SHORTENED and belongs in UNGRADEABLE_SEASONS "
                f"(2020 was 60 games and cleared this floor with 16 hitters), "
                f"or its fetch returned a partial payload — in which case "
                f"delete its cache under {SEASON_LINES_DIR} and re-run "
                f"fetch_mlb_outcomes."
            )
        rated = block["player_type"] == "hitter"
        assert block.loc[rated, "OPS"].notna().all(), (
            f"grade_mlb_seasons: season {season} has a rosterable hitter with a "
            f"missing OPS. A NaN propagates into FV and silently voids the "
            f"whole season — check _parse_splits' rate parsing."
        )
        assert block.loc[~rated, ["ERA", "WHIP"]].notna().all().all(), (
            f"grade_mlb_seasons: season {season} has a rosterable pitcher with "
            f"a missing ERA or WHIP. Those are derived from earnedRuns / hits / "
            f"walks and `outs`, so a NaN means a field went missing upstream."
        )
        scored.append(add_fantasy_value(block)["FV"])
    # add_fantasy_value preserves the input index, so this aligns by row.
    mlb.loc[mlb["rosterable"], "FV"] = pd.concat(scored)
    assert mlb.loc[mlb["rosterable"], "FV"].notna().all(), (
        "grade_mlb_seasons: some rosterable season-lines came back without an "
        "FV. The per-season concat failed to align — check that "
        "add_fantasy_value still returns the input index."
    )

    mlb["fv_rank"] = (
        mlb.loc[mlb["rosterable"]]
        .groupby(["season", "player_type"])["FV"]
        .rank(ascending=False, method="min")
    )
    slots = mlb["player_type"].map(STARTER_SLOTS)
    stars = mlb["player_type"].map(STAR_SLOTS)
    mlb["starter_grade"] = mlb["fv_rank"].notna() & (mlb["fv_rank"] <= slots)
    mlb["star_grade"] = mlb["fv_rank"].notna() & (mlb["fv_rank"] <= stars)
    print(
        f"grade_mlb_seasons: {int(mlb['starter_grade'].sum()):,} starter-grade "
        f"and {int(mlb['star_grade'].sum()):,} star-grade season-lines "
        f"(slots {STARTER_SLOTS}, star cutoffs {STAR_SLOTS})"
    )
    return mlb


def assign_tier(
    reached: pd.Series, n_starter_seasons: pd.Series, n_star_seasons: pd.Series
) -> pd.Series:
    """Map a career summary to one of TIERS. Highest tier wins.

    star     : >= 1 star-grade season
    regular  : >= 2 starter-grade seasons
    fringe   : reached MLB but neither
    never    : no MLB appearance in the outcome window
    """
    tier = pd.Series("never", index=reached.index, dtype=object)
    tier[reached.astype(bool)] = "fringe"
    tier[n_starter_seasons.fillna(0) >= 2] = "regular"
    tier[n_star_seasons.fillna(0) >= 1] = "star"
    return tier


def build_cohort(milb: pd.DataFrame, mlb: pd.DataFrame) -> pd.DataFrame:
    """Join observable minor-league state to its 8-year fantasy outcome.

    Requires: `milb` as returned by fetch_milb_history, `mlb` by
        fetch_mlb_outcomes.
    Returns:
        One row per (player, cohort season, level) with the state columns from
        `add_level_context` plus: reached, first_mlb_season, years_to_mlb,
        arrival, age_at_first_mlb, n_mlb_seasons, n_starter_seasons,
        n_star_seasons, best_FV, tier, and window career totals car_R … car_WHIP.

    Note:
        Rows are dropped when the player had already reached MLB before the
        cohort season (he is a demoted major leaguer, not a prospect), when the
        season is outside COHORT_SEASONS (post-2018 is censored against the
        8-year window), or when playing time is below the level pool gate.
    """
    milb = add_level_context(milb)
    mlb = grade_mlb_seasons(mlb)

    first_mlb_ever = mlb.groupby("player_id")["season"].min()
    milb["first_mlb_ever"] = milb["player_id"].map(first_mlb_ever)

    n0 = len(milb)
    keep = (
        milb["season"].isin(list(COHORT_SEASONS))
        & milb["in_pool"]
        & milb["age"].notna()
        & ~(milb["first_mlb_ever"] < milb["season"])
    )
    cohort = milb.loc[keep].copy()
    print(
        f"build_cohort: {len(cohort):,} cohort rows of {n0:,} "
        f"(seasons {COHORT_SEASONS.start}-{COHORT_SEASONS.stop - 1}, "
        f"playing-time gate, not already in MLB)"
    )
    assert len(cohort) > 0, (
        "build_cohort: no rows survived the cohort filter. Check that "
        "fetch_milb_history covered COHORT_SEASONS and that `in_pool` is set."
    )

    cohort = cohort.reset_index(drop=True)
    cohort["obs_id"] = cohort.index

    # Many-to-many on MLBAM id only, then windowed. NEVER on name.
    cols = ["player_id", "player_type", "season", "age", "PA", "IP", "FV",
            "starter_grade", "star_grade", *HIT_CATS, *PITCH_CATS]
    mlb_side = mlb[cols].rename(columns={"season": "season_mlb", "age": "age_mlb"})
    joined = cohort[["obs_id", "player_id", "player_type", "season"]].rename(
        columns={"season": "season_milb"}
    ).merge(mlb_side, on=["player_id", "player_type"], how="inner")
    in_window = (joined["season_mlb"] >= joined["season_milb"]) & (
        joined["season_mlb"] <= joined["season_milb"] + OUTCOME_WINDOW_YEARS
    )
    joined = joined.loc[in_window].sort_values(["obs_id", "season_mlb"])
    print(
        f"build_cohort: {len(joined):,} in-window MLB season-lines matched to "
        f"{joined['obs_id'].nunique():,} of {len(cohort):,} observations"
    )

    grp = joined.groupby("obs_id")
    summary = pd.DataFrame(
        {
            "first_mlb_season": grp["season_mlb"].min(),
            "n_mlb_seasons": grp["season_mlb"].nunique(),
            "n_starter_seasons": grp["starter_grade"].sum(),
            "n_star_seasons": grp["star_grade"].sum(),
            "best_FV": grp["FV"].max(),
            "car_PA": grp["PA"].sum(),
            "car_IP": grp["IP"].sum(),
        }
    )
    for cat in ("R", "HR", "RBI", "SB", "W", "SV", "K"):
        summary[f"car_{cat}"] = grp[cat].sum()
    # Ratio categories are volume-weighted, never averaged (AGENTS.md).
    for cat, weight in (("OPS", "PA"), ("ERA", "IP"), ("WHIP", "IP")):
        num = joined.assign(_w=joined[cat] * joined[weight]).groupby("obs_id")["_w"].sum()
        summary[f"car_{cat}"] = num / summary[f"car_{weight}"].where(
            summary[f"car_{weight}"] > 0
        )
    # `joined` is sorted by (obs_id, season_mlb), so first() is the debut row.
    summary["age_at_first_mlb"] = grp["age_mlb"].first()

    cohort = cohort.merge(summary, left_on="obs_id", right_index=True, how="left")
    cohort["reached"] = cohort["first_mlb_season"].notna()
    for col in ("n_mlb_seasons", "n_starter_seasons", "n_star_seasons"):
        cohort[col] = cohort[col].fillna(0).astype(int)
    cohort["years_to_mlb"] = (
        cohort["first_mlb_season"] - cohort["season"]
    ).astype("Int64")
    cohort["arrival"] = np.where(
        cohort["reached"],
        cohort["years_to_mlb"].astype("string"),
        f"never_within_{OUTCOME_WINDOW_YEARS}",
    )
    cohort["tier"] = assign_tier(
        cohort["reached"], cohort["n_starter_seasons"], cohort["n_star_seasons"]
    )

    mix = cohort.groupby("player_type")["tier"].value_counts(normalize=True).unstack()
    print("build_cohort: tier mix by player_type\n" + mix.to_string(float_format="%.4f"))
    return cohort


# ---------------------------------------------------------------------------
# Rate tables
# ---------------------------------------------------------------------------


def _rate_table(
    cohort: pd.DataFrame, keys: list[str], outcome: str, outcomes: list[str],
    conditioning: str,
) -> pd.DataFrame:
    """Long-format P(outcome | keys) with n and cell_n on every row.

    No smoothing and no hiding: every cell that exists is emitted with its
    denominator, and cells under MIN_CELL_N are flagged rather than dropped.
    """
    counts = (
        cohort.groupby(keys + [outcome], observed=True)
        .size()
        .unstack(outcome, fill_value=0)
        .reindex(columns=outcomes, fill_value=0)
    )
    cell_n = counts.sum(axis=1).rename("cell_n")
    long = counts.reset_index().melt(id_vars=keys, var_name=outcome, value_name="n")
    long = long.merge(cell_n.reset_index(), on=keys, validate="many_to_one")
    long["p"] = long["n"] / long["cell_n"]
    long["sparse"] = long["cell_n"] < MIN_CELL_N
    long["conditioning"] = conditioning
    return long


def _tabulation_rows(cohort: pd.DataFrame) -> pd.DataFrame:
    """Cohort rows restricted to the ages worth tabulating."""
    rows = cohort.loc[
        cohort["age"].between(PRIOR_AGE_MIN, PRIOR_AGE_MAX)
        & cohort["age_rel_bucket"].notna()
        & cohort["perf_bucket"].notna()
    ].copy()
    rows["age"] = rows["age"].astype(int)
    assert len(rows) > 0, (
        f"_tabulation_rows: no cohort rows with age in "
        f"[{PRIOR_AGE_MIN}, {PRIOR_AGE_MAX}] and both buckets populated. A NaN "
        f"perf_index for every row means the level normalization failed."
    )
    return rows


def build_outcome_rates(
    milb: pd.DataFrame, mlb: pd.DataFrame, cohort: pd.DataFrame | None = None
) -> pd.DataFrame:
    """P(tier | state), long format, both the full and the marginal conditioning.

    Args:
        milb: Minor-league season lines from fetch_milb_history.
        mlb: MLB season lines from fetch_mlb_outcomes.
        cohort: An already-built cohort from `build_cohort`, to avoid redoing
            the join when both rate tables are built in one run.

    Returns:
        Columns: conditioning, player_type, age, sport_id, level,
        age_rel_bucket, perf_bucket, tier, n, cell_n, p, sparse.
        `conditioning == "age_level"` rows are the well-populated marginal and
        carry the literal "ALL" in the two bucket columns;
        `conditioning == "full"` rows condition on all four state variables.
    """
    if cohort is None:
        cohort = build_cohort(milb, mlb)
    rows = _tabulation_rows(cohort)
    base = ["player_type", "age", "sport_id", "level"]
    full = _rate_table(
        rows, base + ["age_rel_bucket", "perf_bucket"], "tier", list(TIERS), "full"
    )
    marginal = _rate_table(rows, base, "tier", list(TIERS), "age_level")
    marginal["age_rel_bucket"] = "ALL"
    marginal["perf_bucket"] = "ALL"
    out = pd.concat([full, marginal], ignore_index=True)
    n_cells = len(full) // len(TIERS)
    n_sparse = int(full.loc[full["tier"] == "never", "sparse"].sum())
    print(
        f"build_outcome_rates: {n_cells:,} fully-conditioned cells, {n_sparse:,} "
        f"of them below n={MIN_CELL_N} (flagged, not smoothed); "
        f"{len(marginal) // len(TIERS):,} marginal cells"
    )
    return out


def build_arrival_hazard(
    milb: pd.DataFrame, mlb: pd.DataFrame, cohort: pd.DataFrame | None = None
) -> pd.DataFrame:
    """P(first reaches MLB t years after this season | age, level), plus never.

    The ETA distribution, not its mean: a prospect who takes 6 years costs twice
    the roster time of one who takes 3, and a valuation model needs the spread.

    Returns:
        Long format with columns: conditioning, player_type, age, sport_id,
        level, arrival, n, cell_n, p, sparse. `arrival` takes the values "0"…
        "8" and "never_within_8", so p sums to 1 per cell INCLUDING the
        never-arrives mass.
    """
    if cohort is None:
        cohort = build_cohort(milb, mlb)
    rows = _tabulation_rows(cohort)
    arrivals = [str(t) for t in range(OUTCOME_WINDOW_YEARS + 1)] + [
        f"never_within_{OUTCOME_WINDOW_YEARS}"
    ]
    out = _rate_table(
        rows, ["player_type", "age", "sport_id", "level"], "arrival", arrivals,
        "age_level",
    )
    ever = rows.groupby(["player_type", "sport_id"])["reached"].mean()
    print(
        "build_arrival_hazard: P(reach MLB within "
        f"{OUTCOME_WINDOW_YEARS}y) by type and level\n"
        + ever.to_string(float_format="%.4f")
    )
    return out


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_outcome_rates() -> pd.DataFrame:
    """Read the outcome-rate table from disk once per process.

    Memoized read of an immutable file — not mutable global state. Call
    `_load_outcome_rates.cache_clear()` after rebuilding within one process.
    """
    assert OUTCOME_RATES_PATH.exists(), (
        f"_load_outcome_rates: {OUTCOME_RATES_PATH} does not exist. Build it "
        f"with `uv run python -m data_prep.prospect_outcomes`."
    )
    return pd.read_parquet(OUTCOME_RATES_PATH)


def prior_from_rates(
    rates: pd.DataFrame,
    age: int,
    level: int,
    age_rel: float,
    perf: float,
    player_type: str,
) -> dict[str, float]:
    """The pure lookup behind `outcome_prior`, against a supplied rate table.

    Split out from `outcome_prior` so the sparsity and sum-to-one behaviour is
    testable against a small synthetic table with no disk and no patching.

    Args:
        rates: A frame shaped like `build_outcome_rates`' output.
        age: Season age at the observation.
        level: MLB StatsAPI sportId (11 AAA, 12 AA, 13 A+, 14 A, 16 R).
        age_rel: Age minus the level-season mean age. Negative is young.
        perf: Level-normalized performance index, 100 = level average.
        player_type: "hitter" or "pitcher".

    Returns:
        {tier: probability} over all of TIERS, summing to 1.0 INCLUDING the
        never-reached tier. A prior summing to less than 1 would silently
        discount every player by the missing mass, so the sum is asserted.

    Note:
        Asserts loudly on an unpopulated or sparse cell rather than returning a
        default. There is no fallback: if the cell is not there, the honest
        answer is the marginal table, and the caller must ask for it explicitly.
    """
    assert player_type in ("hitter", "pitcher"), (
        f"outcome_prior: player_type must be 'hitter' or 'pitcher', got "
        f"{player_type!r}. Hitter and pitcher outcomes are never pooled."
    )
    assert level in LEVELS and level != 1, (
        f"outcome_prior: level must be a minor-league sportId, got {level}. "
        f"Valid: {[k for k in LEVELS if k != 1]} ({LEVELS})."
    )
    arb = bucket_age_rel(pd.Series([float(age_rel)])).iloc[0]
    pb = bucket_perf(pd.Series([float(perf)])).iloc[0]
    cell = rates.loc[
        (rates["conditioning"] == "full")
        & (rates["player_type"] == player_type)
        & (rates["age"] == int(age))
        & (rates["sport_id"] == int(level))
        & (rates["age_rel_bucket"] == arb)
        & (rates["perf_bucket"] == pb)
    ]
    assert len(cell) == len(TIERS), (
        f"outcome_prior: cell (age={age}, level={level}/{LEVELS[level]}, "
        f"age_rel_bucket={arb}, perf_bucket={pb}, {player_type}) has "
        f"{len(cell)} rows, expected {len(TIERS)}. This state was never "
        f"observed in the 2005-2018 cohorts. Query the conditioning=='age_level' "
        f"marginal in {OUTCOME_RATES_PATH.name} instead — do not invent a prior."
    )
    cell_n = int(cell["cell_n"].iloc[0])
    assert cell_n >= MIN_CELL_N, (
        f"outcome_prior: cell (age={age}, level={LEVELS[level]}, "
        f"age_rel_bucket={arb}, perf_bucket={pb}, {player_type}) has only "
        f"n={cell_n}, below MIN_CELL_N={MIN_CELL_N}. Refusing to return a rate "
        f"estimated off {cell_n} players. Use the conditioning=='age_level' "
        f"marginal, which is much better populated."
    )
    prior = {str(t): float(p) for t, p in zip(cell["tier"], cell["p"])}
    total = sum(prior.values())
    assert abs(total - 1.0) < 1e-9, (
        f"outcome_prior: probabilities sum to {total:.9f}, not 1.0, for "
        f"(age={age}, level={LEVELS[level]}, {player_type}). The tier column "
        f"in {OUTCOME_RATES_PATH.name} is missing a tier — rebuild the table."
    )
    return prior


def outcome_prior(
    age: int, level: int, age_rel: float, perf: float, player_type: str
) -> dict[str, float]:
    """P(tier | age, level, age-relative-to-level, perf) from data/priors/.

    Thin wrapper over `prior_from_rates` that supplies the built rate table.
    See `prior_from_rates` for the argument and failure semantics; in
    particular the returned probabilities sum to 1.0 including the
    never-reached tier, and an unpopulated or sparse cell raises.
    """
    return prior_from_rates(
        _load_outcome_rates(), age, level, age_rel, perf, player_type
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_priors(
    cohort_seasons: range = COHORT_SEASONS,
    sport_ids: tuple[int, ...] = DEFAULT_SPORT_IDS,
    mlb_seasons: range = MLB_SEASONS,
) -> None:
    """Fetch (resumably), build all three tables, and write them to data/priors/."""
    milb = fetch_milb_history(cohort_seasons, sport_ids)
    mlb = fetch_mlb_outcomes(mlb_seasons)

    PRIORS_DIR.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort(milb, mlb)
    cohort.to_parquet(COHORT_PATH, index=False)
    print(f"wrote {len(cohort):,} rows -> {COHORT_PATH.relative_to(REPO_ROOT)}")

    rates = build_outcome_rates(milb, mlb, cohort=cohort)
    rates.to_parquet(OUTCOME_RATES_PATH, index=False)
    print(f"wrote {len(rates):,} rows -> {OUTCOME_RATES_PATH.relative_to(REPO_ROOT)}")

    hazard = build_arrival_hazard(milb, mlb, cohort=cohort)
    hazard.to_parquet(ARRIVAL_HAZARD_PATH, index=False)
    print(f"wrote {len(hazard):,} rows -> {ARRIVAL_HAZARD_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_priors()
