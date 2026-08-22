"""
Aging and attrition priors, computed from MLB season lines 1990-2026.

Two tables, and they are the ONLY forward-projection machinery in the valuation
model (see docs/superpowers/specs/2026-08-21-dynasty-objective-design.md §3):

    decay_table     per-category multiplicative year-over-year change, by age
    survival_table  P(still has meaningful playing time in t+k | age, role)

Both are computed here rather than taken from published curves, because every
published curve is denominated in WAR or in FanGraphs' component set, and we
need our own ten roto categories.

FOUR METHOD CHOICES THAT CHANGE THE ANSWER
------------------------------------------
1. POOLED RATIO, NOT MEAN OF RATIOS. A cell's decay factor is
   `Σ w·y / Σ w·x`, not `mean(y/x)`. Ratios of small counting rates explode —
   a player going from 1 SB to 3 SB is a 3.0x observation that would dominate
   any average. The pooled form is the standard delta-method aggregate and is
   stable near zero.

2. LEAGUE-CENTERED. Every rate is divided by its own season's league rate
   before differencing, so league-wide environment shifts cannot masquerade as
   aging. This is not optional here: league SB volume rose 41% in 2023 on a
   rules change, and raw deltas would read that as 27-year-olds rediscovering
   speed.

3. HARMONIC PLAYING-TIME WEIGHTS. `w = 2·v_t·v_{t+1}/(v_t+v_{t+1})` (Tango).
   A pair where either season is short carries little weight, which is what
   keeps a 40-PA September from moving a cell.

4. DECILES COME FROM SEASON t-1, NOT t. Deciling on the same stat you then age
   induces regression to the mean that is indistinguishable from an aging
   effect — the bottom decile "improves" and the top decile "declines" for
   purely statistical reasons. `build_decay_table` computes both and reports the
   gap, so the size of that artifact is visible instead of assumed. The
   decile-stratified table costs three consecutive seasons per observation,
   which excludes rookies; the pooled table uses all pairs and stays the
   primary product.

CENSORING
---------
Base and outcome seasons are capped at `LAST_COMPLETE_SEASON`. 2026 is in
progress, so including it as an outcome year would report every player as
having lost playing time. 2020 is excluded outright and any window spanning it
is dropped.

Survival additionally requires `base + k <= LAST_COMPLETE_SEASON` before a
(player, k) pair enters the DENOMINATOR. Without that, recent cohorts count as
failures at large k purely because the seasons have not happened yet.
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from .raw_io import DATA_DIR

PRIORS_DIR = DATA_DIR / "priors"
HISTORY_DIR = PRIORS_DIR / "history"

_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
_PAGE = 1000

# 2026 is in progress. Any table that reads a partial season as a full one
# reports league-wide collapse.
LAST_COMPLETE_SEASON: int = 2025
FIRST_SEASON: int = 1990
# The shortened season. Rates are unreliable and any pair spanning it compares
# a 60-game season to a 162-game one.
EXCLUDED_SEASONS: frozenset[int] = frozenset({2020})

# Volume floors for a season to count as a real sample in the decay tables.
MIN_PA: float = 300.0
MIN_IP_START: float = 100.0
MIN_G_RELIEF: int = 30

# Reference volumes. Counting stats are expressed per these so that rate and
# volume age separately — SB rate is nearly flat ages 22-26 while PA attrition
# is steepest for speed profiles, and a rate-only curve overvalues speed.
PER_PA: float = 600.0
PER_IP: float = 200.0
PER_G: float = 65.0

# A starter is a pitcher who mostly starts. Relievers are aged separately
# because SV is driven by role, not skill.
STARTER_GS_SHARE: float = 0.5

# Ages outside this band have too few player-seasons to estimate anything.
MIN_AGE: int = 20
MAX_AGE: int = 40

N_DECILES: int = 10

# Output schemas. Named so an empty result still carries the right columns —
# a bare DataFrame() would fail on column access far from the cause.
DECAY_COLUMNS: tuple[str, ...] = (
    "role", "category", "age", "decile", "variant", "n", "factor", "league_factor",
)
SURVIVAL_COLUMNS: tuple[str, ...] = (
    "role", "age", "k", "n", "rate", "base_floor", "later_floor",
)

# Categories, and how each is built from raw counting columns. Computing from
# counts avoids StatsAPI's string rate fields and their '.---' / '-.--'
# missing sentinels entirely.
HITTER_RATE_CATEGORIES: tuple[str, ...] = ("R", "HR", "RBI", "SB")
PITCHER_RATE_CATEGORIES: tuple[str, ...] = ("W", "K", "SV")
# Ratio categories are already rates; they are not scaled by reference volume.
HITTER_RATIO_CATEGORIES: tuple[str, ...] = ("OPS",)
PITCHER_RATIO_CATEGORIES: tuple[str, ...] = ("ERA", "WHIP")
# Volume ages on its own and is the second half of every counting projection.
VOLUME_CATEGORIES: tuple[str, ...] = ("VOL",)

# Higher is worse for these, so a factor above 1.0 is decline, not improvement.
LOWER_IS_BETTER: frozenset[str] = frozenset({"ERA", "WHIP"})


# ==========================================================================
# FETCH
# ==========================================================================


def _season_path(group: str, season: int) -> Path:
    return HISTORY_DIR / f"{group}_{season}.parquet"


def _fetch_one_season(group: str, season: int) -> pd.DataFrame:
    """Pull every player's season line for one season from StatsAPI.

    `playerPool=all` is mandatory: without it the endpoint silently returns
    only ~140 qualified players instead of ~900.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        response = requests.get(
            _STATS_URL,
            params={
                "stats": "season",
                "group": group,
                "season": season,
                "sportId": 1,
                "playerPool": "all",
                "limit": _PAGE,
                "offset": offset,
            },
            timeout=60,
        )
        assert response.status_code == 200, (
            f"_fetch_one_season: StatsAPI returned {response.status_code} for "
            f"{group} {season}. Check the URL and that sportId=1 is valid."
        )
        payload = response.json()
        splits = [s for g in payload.get("stats", []) for s in g.get("splits", [])]
        if not splits:
            break
        for split in splits:
            row = dict(split["stat"])
            row["playerId"] = int(split["player"]["id"])
            row["name"] = split["player"]["fullName"]
            row["season"] = season
            rows.append(row)
        offset += _PAGE
        total = payload.get("stats", [{}])[0].get("totalSplits")
        if total is None or offset >= int(total):
            break

    assert rows, (
        f"_fetch_one_season: no {group} splits for {season}. An empty response "
        f"means the query is wrong, not that nobody played — check playerPool."
    )
    return pd.DataFrame(rows)


def fetch_history(
    group: str, seasons: range | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Season lines for every player, cached one parquet per season.

    Args:
        group: "hitting" or "pitching".
        seasons: Seasons to cover. Defaults to FIRST_SEASON..current year.
        refresh: Re-fetch seasons already cached. The current season always
            re-fetches regardless, since it is still accumulating.

    Returns:
        One row per player-season. Always carries: playerId, name, season, age,
        plus that group's raw StatsAPI counting columns.
    """
    assert group in ("hitting", "pitching"), (
        f"fetch_history: group must be 'hitting' or 'pitching', got {group!r}."
    )
    if seasons is None:
        seasons = range(FIRST_SEASON, datetime.date.today().year + 1)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    current = datetime.date.today().year

    frames = []
    for season in tqdm(list(seasons), desc=f"history {group}"):
        path = _season_path(group, season)
        if path.exists() and not refresh and season != current:
            frames.append(pd.read_parquet(path))
            continue
        frame = _fetch_one_season(group, season)
        frame.to_parquet(path, index=False)
        frames.append(frame)

    history = pd.concat(frames, ignore_index=True)
    assert not history.duplicated(["playerId", "season"]).any(), (
        "fetch_history: duplicate (playerId, season) rows. StatsAPI should "
        "aggregate traded players across teams; a duplicate means the response "
        "was split by team and every rate below would double-count."
    )
    print(
        f"history {group}: {len(history)} player-seasons, "
        f"{history['season'].min()}-{history['season'].max()}"
    )
    return history


def load_history(group: str) -> pd.DataFrame:
    """Read the cached history, whether stored per-season or as one file.

    Fails loudly rather than silently fetching: a caller that wanted the
    network should call `fetch_history`.
    """
    combined = HISTORY_DIR / f"{group}.parquet"
    per_season = sorted(HISTORY_DIR.glob(f"{group}_*.parquet"))
    assert combined.exists() or per_season, (
        f"load_history: no cached {group} history in {HISTORY_DIR}. Run "
        f"`fetch_history({group!r})` first — it caches one parquet per season."
    )
    if per_season:
        return pd.concat(
            [pd.read_parquet(p) for p in per_season], ignore_index=True
        )
    return pd.read_parquet(combined)


# ==========================================================================
# NORMALISE
# ==========================================================================


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numeric view of a raw StatsAPI column, absent columns becoming 0.

    StatsAPI omits keys rather than sending nulls, and writes rate fields as
    strings with '.---' as its missing sentinel. Everything here is built from
    counting columns, so coercion failures are genuinely zero.
    """
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def prepare_hitters(history: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw hitting lines to (id, season, age, volume, our categories).

    Adds columns: playerId, season, age, VOL, R, HR, RBI, SB, OPS, role.
    Counting stats stay as TOTALS here; `_rate_columns` scales them per 600 PA.

    OPS is computed from counts (OBP = (H+BB+HBP)/(AB+BB+HBP+SF),
    SLG = TB/AB) rather than parsed from the string field, so there is no
    sentinel handling and no silent coercion.
    """
    pa = _num(history, "plateAppearances")
    ab = _num(history, "atBats")
    bb = _num(history, "baseOnBalls")
    hbp = _num(history, "hitByPitch")
    sf = _num(history, "sacFlies")
    hits = _num(history, "hits")
    on_base_denominator = (ab + bb + hbp + sf).replace(0.0, np.nan)

    frame = pd.DataFrame(
        {
            "playerId": history["playerId"].astype(int),
            "season": history["season"].astype(int),
            "age": pd.to_numeric(history["age"], errors="coerce"),
            "VOL": pa,
            "R": _num(history, "runs"),
            "HR": _num(history, "homeRuns"),
            "RBI": _num(history, "rbi"),
            "SB": _num(history, "stolenBases"),
        }
    )
    obp = (hits + bb + hbp) / on_base_denominator
    slg = _num(history, "totalBases") / ab.replace(0.0, np.nan)
    frame["OPS"] = (obp + slg).astype(float)
    frame["role"] = "hitter"
    return frame.dropna(subset=["age"])


def prepare_pitchers(history: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw pitching lines to (id, season, age, volume, our categories).

    Adds columns: playerId, season, age, VOL, W, K, SV, ERA, WHIP, role, games.

    `role` splits starters from relievers at STARTER_GS_SHARE of appearances.
    They must not be pooled: SV is a role outcome, and a starter's volume unit
    is innings while a reliever's is appearances.

    IP is derived from `outs`, never from `inningsPitched` — that field is
    "76.1" meaning 76 AND ONE THIRD, and reading it as a float understates
    every rate denominator.
    """
    outs = _num(history, "outs")
    innings = outs / 3.0
    games = _num(history, "gamesPitched")
    starts = _num(history, "gamesStarted")

    frame = pd.DataFrame(
        {
            "playerId": history["playerId"].astype(int),
            "season": history["season"].astype(int),
            "age": pd.to_numeric(history["age"], errors="coerce"),
            "VOL": innings,
            "games": games,
            "W": _num(history, "wins"),
            "K": _num(history, "strikeOuts"),
            "SV": _num(history, "saves"),
        }
    )
    safe_innings = innings.replace(0.0, np.nan)
    frame["ERA"] = 9.0 * _num(history, "earnedRuns") / safe_innings
    frame["WHIP"] = (_num(history, "hits") + _num(history, "baseOnBalls")) / safe_innings
    share = np.where(games > 0, starts / games.replace(0.0, np.nan), 0.0)
    frame["role"] = np.where(share >= STARTER_GS_SHARE, "starter", "reliever")
    return frame.dropna(subset=["age"])


def _qualified(frame: pd.DataFrame) -> pd.Series:
    """Rows with enough volume for their role to be a real sample."""
    is_hitter = frame["role"] == "hitter"
    is_starter = frame["role"] == "starter"
    is_reliever = frame["role"] == "reliever"
    return (
        (is_hitter & (frame["VOL"] >= MIN_PA))
        | (is_starter & (frame["VOL"] >= MIN_IP_START))
        | (is_reliever & (frame.get("games", pd.Series(0, index=frame.index)) >= MIN_G_RELIEF))
    )


def _reference_volume(role: str) -> float:
    if role == "hitter":
        return PER_PA
    if role == "starter":
        return PER_IP
    return PER_G


def _rate_columns(frame: pd.DataFrame, categories: tuple[str, ...]) -> pd.DataFrame:
    """Express counting categories per reference volume, in place on a copy.

    Relievers are scaled per appearance, not per inning: SV per 65 G is the
    meaningful unit and SV per 200 IP is not.
    """
    frame = frame.copy()
    denominator = frame["VOL"].copy()
    is_reliever = frame["role"] == "reliever"
    if "games" in frame.columns:
        denominator = denominator.where(~is_reliever, frame["games"])
    reference = frame["role"].map(_reference_volume).astype(float)
    safe = denominator.replace(0.0, np.nan)
    for category in categories:
        frame[category] = frame[category] / safe * reference
    return frame


def _league_rates(
    frame: pd.DataFrame, categories: tuple[str, ...]
) -> pd.DataFrame:
    """Per-(season, role) league rate for each category, over qualified rows.

    Volume-weighted, i.e. pooled across players rather than averaged over
    them. An unweighted mean would let a 300-PA bench bat count as much as a
    700-PA regular in setting the league baseline.
    """
    qualified = frame[_qualified(frame)].copy()
    assert len(qualified) > 0, (
        "_league_rates: no rows clear the volume floors. Check MIN_PA / "
        "MIN_IP_START / MIN_G_RELIEF against the units in VOL."
    )
    weight = qualified["VOL"].where(
        qualified["role"] != "reliever",
        qualified.get("games", qualified["VOL"]),
    )
    qualified["_w"] = weight
    out = []
    for (season, role), block in qualified.groupby(["season", "role"], sort=True):
        row = {"season": season, "role": role, "n": len(block)}
        total_weight = block["_w"].sum()
        for category in categories:
            values = block[category]
            usable = values.notna()
            row[category] = (
                float((values[usable] * block.loc[usable, "_w"]).sum())
                / float(block.loc[usable, "_w"].sum())
                if usable.any() and block.loc[usable, "_w"].sum() > 0
                else np.nan
            )
        row["_total_weight"] = float(total_weight)
        out.append(row)
    return pd.DataFrame(out)


# ==========================================================================
# DECAY
# ==========================================================================


def _consecutive_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Join each player-season to the same player's next season.

    Returns one row per (player, t) with `_next` suffixed columns for t+1.
    Pairs spanning an excluded season, or reaching past
    LAST_COMPLETE_SEASON, are dropped — a partial 2026 read as a full season
    would report league-wide collapse.
    """
    left = frame.copy()
    right = frame.copy()
    right["season"] = right["season"] - 1
    pairs = left.merge(
        right, on=["playerId", "season"], suffixes=("", "_next"), how="inner"
    )
    pairs = pairs[
        (~pairs["season"].isin(EXCLUDED_SEASONS))
        & (~(pairs["season"] + 1).isin(EXCLUDED_SEASONS))
        & (pairs["season"] + 1 <= LAST_COMPLETE_SEASON)
    ]
    return pairs.reset_index(drop=True)


def _harmonic_weight(a: pd.Series, b: pd.Series) -> pd.Series:
    """Harmonic mean of two playing-time values; 0 when either is 0."""
    total = a + b
    return np.where(total > 0, 2.0 * a * b / total.replace(0.0, np.nan), 0.0)


def _assign_deciles(frame: pd.DataFrame, stat: str) -> pd.Series:
    """Within-(season, role) decile on `stat`, 0 = worst, over qualified rows.

    NaN for unqualified rows. For ERA/WHIP the ranking is inverted so 9 is
    always the best decile regardless of category polarity.
    """
    deciles = pd.Series(np.nan, index=frame.index)
    qualified = _qualified(frame)
    ascending = stat not in LOWER_IS_BETTER
    for _, block in frame[qualified].groupby(["season", "role"], sort=False):
        values = block[stat]
        usable = values.notna()
        if usable.sum() < N_DECILES:
            continue
        ranked = values[usable].rank(pct=True, ascending=ascending)
        deciles.loc[ranked.index] = np.minimum(
            (ranked * N_DECILES).astype(int), N_DECILES - 1
        )
    return deciles


def build_decay_table(
    frame: pd.DataFrame,
    categories: tuple[str, ...],
    decile_stat: str | None = None,
    decile_lag: int = 1,
) -> pd.DataFrame:
    """Multiplicative year-over-year change per (role, category, age).

    Args:
        frame: Output of `prepare_hitters` or `prepare_pitchers`, with counting
            categories already scaled by `_rate_columns`.
        categories: Category columns to build curves for.
        decile_stat: If given, stratify by that stat's decile.
        decile_lag: Seasons before the pair's base season to measure the decile
            in. 1 (default) is the honest choice: measuring in t-1 keeps the
            stratification off the same observation being aged. 0 reproduces
            the naive version, whose apparent aging is contaminated by
            regression to the mean; it exists only so the size of that
            artifact can be measured by differencing the two.

    Returns:
        Long frame: role, category, age, decile, variant, n, factor,
        league_factor. `decile` is -1 for the unstratified curve. `variant` is
        "strict" (both seasons qualified) or "inclusive" (only season t
        qualified, so collapses are retained). `factor` > 1 means the stat rose;
        for ERA/WHIP that is decline, not improvement.
    """
    assert "VOL" in frame.columns, (
        "build_decay_table: frame needs a VOL column. Pass the output of "
        "prepare_hitters/prepare_pitchers, which sets it."
    )
    league = _league_rates(frame, categories)
    pairs = _consecutive_pairs(frame)
    # An empty result is a legitimate outcome of correct filtering (every pair
    # spanned 2020, or ended in the in-progress season). Asserting here cannot
    # tell that apart from a broken filter, so the non-empty check belongs to
    # the caller, which knows whether emptiness is possible. See build_all.
    if pairs.empty:
        print("decay table: no surviving season pairs; returning empty")
        return pd.DataFrame(columns=list(DECAY_COLUMNS))

    if decile_stat is not None:
        assert decile_stat in frame.columns, (
            f"build_decay_table: decile_stat {decile_stat!r} is not a column. "
            f"Available: {sorted(frame.columns)}."
        )
        prior = frame[["playerId", "season"]].copy()
        prior["_decile"] = _assign_deciles(frame, decile_stat)
        # Shift forward by the lag so the merge lands on the pair's base
        # season: a decile measured in season s attaches to the pair based at
        # s + decile_lag.
        prior["season"] = prior["season"] + decile_lag
        pairs = pairs.merge(prior, on=["playerId", "season"], how="inner")
        pairs = pairs[pairs["_decile"].notna()]
    else:
        pairs["_decile"] = -1.0

    reliever = pairs["role"] == "reliever"
    volume_now = pairs["VOL"].where(~reliever, pairs.get("games", pairs["VOL"]))
    volume_next = pairs["VOL_next"].where(
        ~reliever, pairs.get("games_next", pairs["VOL_next"])
    )
    pairs["_w"] = _harmonic_weight(volume_now, volume_next)
    pairs["_qual_now"] = _qualified(pairs).values
    # Select ONLY the _next columns before stripping the suffix. Renaming in
    # place would collide with the unsuffixed originals and leave two columns
    # named VOL, which silently breaks every lookup downstream.
    next_columns = {c: c[: -len("_next")] for c in pairs.columns if c.endswith("_next")}
    assert "role_next" in pairs.columns and "VOL_next" in pairs.columns, (
        "build_decay_table: expected role_next and VOL_next from the "
        "self-merge. A player's role can change between seasons, so the "
        "qualification test for t+1 must use t+1's own role and volume."
    )
    next_frame = pairs[list(next_columns)].rename(columns=next_columns)
    pairs["_qual_next"] = _qualified(next_frame).values

    rows = []
    for category in categories:
        merged = pairs.merge(
            league[["season", "role", category]].rename(
                columns={category: "_L_now"}
            ),
            on=["season", "role"],
            how="left",
        )
        merged["_season_next"] = merged["season"] + 1
        merged = merged.merge(
            league[["season", "role", category]].rename(
                columns={"season": "_season_next", category: "_L_next"}
            ),
            on=["_season_next", "role"],
            how="left",
        )
        for variant in ("strict", "inclusive"):
            usable = merged["_qual_now"] & (
                merged["_qual_next"] if variant == "strict" else True
            )
            block = merged[
                usable
                & merged[category].notna()
                & merged[f"{category}_next"].notna()
                & merged["_L_now"].notna()
                & merged["_L_next"].notna()
                & (merged["_w"] > 0)
            ]
            for (role, age, decile), cell in block.groupby(
                ["role", "age", "_decile"], sort=True
            ):
                if not (MIN_AGE <= age <= MAX_AGE):
                    continue
                x = (cell[category] / cell["_L_now"] * cell["_w"]).sum()
                y = (cell[f"{category}_next"] / cell["_L_next"] * cell["_w"]).sum()
                if x <= 0:
                    continue
                rows.append(
                    {
                        "role": role,
                        "category": category,
                        "age": int(age),
                        "decile": int(decile),
                        "variant": variant,
                        "n": len(cell),
                        "factor": float(y / x),
                        "league_factor": float(
                            (cell["_L_next"] / cell["_L_now"]).mean()
                        ),
                    }
                )

    table = pd.DataFrame(rows, columns=list(DECAY_COLUMNS))
    if table.empty:
        print("decay table: no cells cleared the volume/age filters")
        return table
    print(
        f"decay table: {len(table)} cells, "
        f"{table['category'].nunique()} categories, "
        f"ages {table['age'].min()}-{table['age'].max()}"
    )
    return table.sort_values(
        ["role", "category", "variant", "decile", "age"]
    ).reset_index(drop=True)


# ==========================================================================
# SURVIVAL
# ==========================================================================


def build_survival_table(
    frame: pd.DataFrame,
    base_floor: float,
    later_floor: float,
    max_k: int = 8,
    volume_column: str = "VOL",
) -> pd.DataFrame:
    """P(volume >= later_floor in t+k | volume >= base_floor at age A).

    Args:
        frame: Output of `prepare_hitters` or `prepare_pitchers`.
        base_floor: Volume the base season must clear.
        later_floor: Volume the t+k season must clear to count as survival.
        max_k: Horizons to compute, 1..max_k.
        volume_column: "VOL" for PA/IP, "games" for reliever appearances.

    Returns:
        Long frame: role, age, k, n, rate, base_floor, later_floor.
        `n` is the CENSORING-CORRECTED denominator: a (player, k) pair only
        enters it when season base+k has actually been played. Without that
        correction recent cohorts read as failures at large k because the
        seasons do not exist yet, which deflates every long-horizon rate.
    """
    assert volume_column in frame.columns, (
        f"build_survival_table: no '{volume_column}' column. Use 'VOL' for "
        f"PA/IP or 'games' for reliever appearances."
    )
    usable = frame[~frame["season"].isin(EXCLUDED_SEASONS)].copy()
    volume = usable[volume_column]
    base = usable[
        (volume >= base_floor)
        & (usable["season"] <= LAST_COMPLETE_SEASON)
        & usable["age"].between(MIN_AGE, MAX_AGE)
    ][["playerId", "season", "age", "role"]].copy()

    reached = usable[usable[volume_column] >= later_floor][
        ["playerId", "season"]
    ].copy()
    reached["_hit"] = True

    rows = []
    for k in range(1, max_k + 1):
        candidates = base[base["season"] + k <= LAST_COMPLETE_SEASON].copy()
        if (base["season"] + k).isin(EXCLUDED_SEASONS).any():
            candidates = candidates[
                ~(candidates["season"] + k).isin(EXCLUDED_SEASONS)
            ]
        if candidates.empty:
            continue
        candidates["season"] = candidates["season"] + k
        merged = candidates.merge(reached, on=["playerId", "season"], how="left")
        merged["_hit"] = merged["_hit"].fillna(False)
        for (role, age), cell in merged.groupby(["role", "age"], sort=True):
            rows.append(
                {
                    "role": role,
                    "age": int(age),
                    "k": k,
                    "n": len(cell),
                    "rate": float(cell["_hit"].mean()),
                    "base_floor": base_floor,
                    "later_floor": later_floor,
                }
            )

    table = pd.DataFrame(rows, columns=list(SURVIVAL_COLUMNS))
    if table.empty:
        print(
            f"survival table: no observable cells for base_floor={base_floor} "
            f"(every cohort censored, or nobody cleared the floor)"
        )
        return table
    print(
        f"survival table: {len(table)} cells, k=1..{table['k'].max()}, "
        f"ages {table['age'].min()}-{table['age'].max()}"
    )
    return table.sort_values(["role", "age", "k"]).reset_index(drop=True)


# ==========================================================================
# LOOKUPS — what the valuation code calls
# ==========================================================================


def cumulative_decay(
    decay: pd.DataFrame,
    age: int,
    horizon: int,
    category: str,
    role: str,
    decile: int = -1,
    variant: str = "inclusive",
) -> float:
    """Compound the one-year factors from `age` forward `horizon` years.

    Two boundaries, handled differently on purpose:

    * `age < MIN_AGE` ASSERTS. There is no MLB data below 20, so any factor
      would be invented. A teenage prospect must never be aged from his
      current age — his line belongs at his projected ARRIVAL age (from the
      outcome mixture) and is aged forward from there. Hitting this assert
      means the caller aged the wrong starting point.
    * `age + horizon > MAX_AGE` returns 0.0. Past the fitted band we declare
      the player finished rather than extrapolate. This is a real assumption,
      but a small one: measured survival for age 36-40 hitters at k=5 is 4.4%,
      so the discarded value is close to zero already.

    Returns 1.0 for horizon 0.
    """
    assert horizon >= 0, f"cumulative_decay: horizon must be >= 0, got {horizon}."
    assert age >= MIN_AGE, (
        f"cumulative_decay: age {age} is below the fitted band's floor "
        f"{MIN_AGE}. No MLB player-seasons exist there, so every factor would "
        f"be invented. Age a prospect from his projected ARRIVAL age, not from "
        f"his current age."
    )
    if age > MAX_AGE or age + horizon > MAX_AGE:
        return 0.0
    subset = decay[
        (decay["category"] == category)
        & (decay["role"] == role)
        & (decay["variant"] == variant)
        & (decay["decile"] == decile)
    ].set_index("age")["factor"]
    assert not subset.empty, (
        f"cumulative_decay: no cells for category={category!r} role={role!r} "
        f"variant={variant!r} decile={decile}. Check build_decay_table was run "
        f"for this category, and that decile=-1 is the unstratified curve."
    )
    total = 1.0
    for step in range(horizon):
        current = age + step
        assert current in subset.index, (
            f"cumulative_decay: no factor at age {current} for {category}/"
            f"{role}/decile {decile}. The cell is empty — either the age band "
            f"has a hole or the volume floors excluded everyone there."
        )
        total *= float(subset.loc[current])
    return total


def survival_factor(
    survival: pd.DataFrame, age: int, horizon: int, role: str
) -> float:
    """P(still has playing time `horizon` years out), 1.0 at horizon 0.

    Reads the table directly rather than compounding one-year rates: the table
    already measures t+k against the base season, and compounding would
    double-count the players who lose and regain a job.
    """
    assert horizon >= 0, f"survival_factor: horizon must be >= 0, got {horizon}."
    if horizon == 0:
        return 1.0
    subset = survival[
        (survival["role"] == role)
        & (survival["age"] == age)
        & (survival["k"] == horizon)
    ]
    assert len(subset) == 1, (
        f"survival_factor: expected exactly one cell for role={role!r} "
        f"age={age} k={horizon}, found {len(subset)}. An empty cell means the "
        f"cohort was censored or below the age band; do not substitute a "
        f"neighbouring age silently."
    )
    return float(subset["rate"].iloc[0])


# ==========================================================================
# BUILD
# ==========================================================================


def build_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Build every prior table and write them to data/priors/.

    Returns a dict of the tables, keyed by output filename stem.
    """
    PRIORS_DIR.mkdir(parents=True, exist_ok=True)
    hitting = fetch_history("hitting", refresh=refresh) if refresh else load_history(
        "hitting"
    )
    pitching = fetch_history("pitching", refresh=refresh) if refresh else load_history(
        "pitching"
    )

    hitters = _rate_columns(prepare_hitters(hitting), HITTER_RATE_CATEGORIES)
    pitchers = _rate_columns(prepare_pitchers(pitching), PITCHER_RATE_CATEGORIES)

    hitter_categories = (
        *HITTER_RATE_CATEGORIES,
        *HITTER_RATIO_CATEGORIES,
        *VOLUME_CATEGORIES,
    )
    pitcher_categories = (
        *PITCHER_RATE_CATEGORIES,
        *PITCHER_RATIO_CATEGORIES,
        *VOLUME_CATEGORIES,
    )

    tables: dict[str, pd.DataFrame] = {}
    tables["decay_hitters"] = build_decay_table(hitters, hitter_categories)
    tables["decay_pitchers"] = build_decay_table(pitchers, pitcher_categories)
    tables["decay_hitters_decile"] = build_decay_table(
        hitters, hitter_categories, decile_stat="OPS"
    )
    tables["decay_pitchers_decile"] = build_decay_table(
        pitchers, pitcher_categories, decile_stat="ERA"
    )
    # The naive stratification, kept ONLY as a diagnostic: deciling on the same
    # season being aged is the regression-to-the-mean artifact the module
    # docstring warns about. Differencing it against the lag-1 table above is
    # how we measure the artifact instead of assuming it away.
    tables["decay_hitters_decile_naive"] = build_decay_table(
        hitters, hitter_categories, decile_stat="OPS", decile_lag=0
    )
    tables["decay_pitchers_decile_naive"] = build_decay_table(
        pitchers, pitcher_categories, decile_stat="ERA", decile_lag=0
    )

    tables["survival_hitters"] = build_survival_table(hitters, 500.0, 400.0)
    tables["survival_hitters_loose"] = build_survival_table(hitters, 500.0, 300.0)
    tables["survival_starters"] = build_survival_table(
        pitchers[pitchers["role"] == "starter"], 150.0, 120.0
    )
    tables["survival_relievers"] = build_survival_table(
        pitchers[pitchers["role"] == "reliever"],
        50.0,
        40.0,
        volume_column="games",
    )

    for name, table in tables.items():
        # Emptiness is a valid outcome for a filtered subset but never for a
        # full production build: it means a floor, a season bound or a category
        # name is wrong. This is the caller that knows that, so the check lives
        # here rather than inside the builders.
        assert not table.empty, (
            f"build_all: table {name!r} came out empty. In a full build every "
            f"table must have cells — check the volume floors (MIN_PA, "
            f"MIN_IP_START, MIN_G_RELIEF), the season bounds "
            f"(FIRST_SEASON..LAST_COMPLETE_SEASON) and that the category names "
            f"match the prepared frame's columns."
        )
        path = PRIORS_DIR / f"{name}.parquet"
        table.to_parquet(path, index=False)
        print(f"  wrote {len(table)} rows -> {path.relative_to(DATA_DIR.parent)}")
    return tables


if __name__ == "__main__":
    build_all()
