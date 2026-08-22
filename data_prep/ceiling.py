"""Rank dynasty players by CEILING, not by expected value.

Why this exists, and why it is not `add_mew`
--------------------------------------------
In a 7-team roto league replacement level is absurdly high: 7 x 28 = 196 roster
spots against ~1,400 major leaguers, so the waiver wire is full of competent
regulars. A player whose MEDIAN outcome is "solid regular" is worth exactly
nothing, because that outcome is free. Only star outcomes move a category.
Every ranking in this module is therefore a ranking of the TAIL, and the whole
design question is "what is this player's 90th percentile", never "what is his
mean".

Three tiers, in increasing order of how much we trust them as tail evidence:

  Tier 1 — talent level, defence stripped out.
      OOPSY's projected career-PEAK season from FanGraphs
      (`type=oopsypeak`), run through `optimizer.player_scoring.
      add_fantasy_value`. We use OOPSY's OFFENSIVE / pitching category line
      and deliberately IGNORE its WAR column: WAR bundles defensive value,
      which is worth zero in this league and which systematically over-credits
      catchers and premium up-the-middle defenders. Peak lines are already
      normalised to 600 PA / 198 IP (70 IP for relievers) and park-neutral,
      so no volume adjustment is wanted or applied.

  Tier 2 — physical ceiling, MEASURED rather than scouted.
      Tier 1 is still a mean, so it cannot be the tail proxy. Tools are the
      hard constraint on the tail: a hitter swinging 77.8 mph can turn into a
      38-homer bat, one at 69.5 mph cannot, whatever his minor-league slash
      line says. Sourced from Baseball Savant's public leaderboards (no auth).

      MEASURED AGREEMENT WITH OOPSY'S OWN FANTASY RANKS (2026-08-21). OOPSY
      publishes "OBP Rank" and "BA Rank" alongside WAR in its midseason top-100
      write-up — its own defence-stripped, standard-scores fantasy ranking. Over
      the 73 hitters on that list that join to the peak feed, Spearman ρ between
      `tier1_FV` and OOPSY's OBP Rank is +0.62. That is agreement, not
      equivalence, and the gap is explained rather than mysterious: this module
      applies NO positional adjustment or replacement level, whereas an auction-
      calculator ranking does, and ρ over a list already truncated to the top
      100 is heavily range-attenuated. Two hypotheses were tested and rejected
      as the cause — restricting the z-score population to MLB-quality hitters
      (wRC+ >= 100, or top 400) moves ρ to 0.57-0.59, and substituting OBP for
      OPS as the rate category moves it to 0.64. Worth re-checking if it ever
      drops below ~0.5. Note also that within that top-100 list OOPSY's WAR rank
      and its OBP rank agree at ρ = +0.91, so for PROSPECTS specifically the
      defence contamination in WAR is much smaller than it is for major leaguers
      (catchers and shortstops with real defensive value); the WAR column is
      still ignored here, but that is the weaker of the two reasons.

  Tier 3 — an explicit distribution.
      Baseball Prospectus PECOTA publishes 10th/50th/90th percentile lines and
      the 90th is literally the number we want. It is paywalled, so this module
      ships the hook (`load_pecota_tail`) and a clean not-configured path. With
      no PECOTA the combined score is an APPROXIMATION of the tail built from
      Tier 1 + Tier 2, and says so in the column comment and the printed
      header.

Percentiles are OURS, computed from raw values
---------------------------------------------
Savant renders percentile sliders on its player pages, and scraping them has
proven unreliable — two fetches of the same page returned contradictory
percentiles for the same player. So this module never reads a published
percentile. Every `pct_*` column here is `rank(pct=True)` over the raw metric
within the relevant population (that season's leaderboard, split by
player_type), which is reproducible from the snapshot on disk.

Auth
----
The peak feed comes from the FanGraphs JSON API and needs a MEMBER browser
session (`get_fangraphs_session`). When that cookie expires the fetch fails
loudly with re-login instructions rather than falling back to something
plausible. `data_prep/wayback.py` harvests the server-rendered /projections
page instead of the API and needs no cookie at all; it is the escape hatch when
the cookie has expired and cannot be refreshed, but it is not wired in here
because it only carries whatever feeds the Archive happened to capture.

Usage:
    uv run python -m data_prep.ceiling --top 40
    uv run python -m data_prep.ceiling --free-agents-only --minors-only
    uv run python -m data_prep.ceiling --refresh --fv-bar 2.5
"""

import argparse
import datetime
import io

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from optimizer.config import (
    FANTRAX_TEAM_IDS,
    HITTING_SLOTS,
    MY_TEAM_NAME,
    PITCHING_SLOTS,
    TEAM_ID_TO_NAME,
)
from optimizer.player_scoring import add_fantasy_value
from optimizer.players import get_eligible_slots

from .build import _team_key, match_rows
from .names import normalize_name, strip_name_suffix
from .raw_io import available_dates, read_latest_raw, write_raw

# ── Tier 1 ────────────────────────────────────────────────────────────────

# Logical name of the peak feed, registered in scrape_fangraphs.PROJECTION_TYPES
# (which maps it to the FanGraphs API `type` string). VERIFIED against the live
# API on 2026-08-21: type=oopsypeak returns 2,865 bat rows and 4,256 pit rows.
PEAK_SYSTEM: str = "oopsypeak"

# The RoS system SV is borrowed from — see `_merge_role_saves` for why.
SAVES_SYSTEM: str = "steamer"

# Peak-feed columns kept, per side. Deliberately NOT reusing
# scrape_fangraphs.HITTER_STAT_COLUMNS: that path requires a non-null FanGraphs
# PlayerId because it feeds the Ottoneu market join, and the peak feed has a row
# without one. Ceiling work joins on MLBAMID (Savant) and on name (Fantrax), so
# PlayerId is not a key here and demanding it would fail the fetch over a column
# nothing downstream reads. WAR is excluded on purpose — see add_tier1_score.
_PEAK_SHARED: list[str] = ["Name", "Team", "player_type", "MLBAMID"]
_PEAK_HITTER: list[str] = ["PA", "R", "HR", "RBI", "SB", "OPS", "wRC+"]
_PEAK_PITCHER: list[str] = ["IP", "GS", "W", "SV", "SO", "ERA", "WHIP"]


def fetch_peak_snapshot() -> pd.DataFrame:
    """Fetch the OOPSY peak feed (both sides) into one raw snapshot frame.

    Reuses `scrape_fangraphs.fetch_projections` for auth and the API call, so
    the member-cookie assertion and its re-login instructions live in exactly
    one place.

    Returns:
        DataFrame with _PEAK_SHARED + _PEAK_HITTER + _PEAK_PITCHER. Each row
        carries values only for its own side; the other side's columns are NULL,
        not zero — zero-filling is the scoring step's decision.
    """
    from .scrape_fangraphs import (
        API_RENAMES,
        fetch_projections,
        get_fangraphs_session,
    )

    session = get_fangraphs_session()
    sides = []
    for stats, player_type, stat_columns in (
        ("bat", "hitter", _PEAK_HITTER),
        ("pit", "pitcher", _PEAK_PITCHER),
    ):
        frame = pd.DataFrame(
            fetch_projections(session, PEAK_SYSTEM, stats)
        ).rename(columns=API_RENAMES)
        frame["player_type"] = player_type
        columns = _PEAK_SHARED + stat_columns
        missing = [c for c in columns if c not in frame.columns]
        assert not missing, (
            f"OOPSY peak {stats} feed is missing column(s) {missing}.\n"
            f"  Columns returned: {sorted(frame.columns)}\n"
            f"FanGraphs renamed a field — update _PEAK_* in data_prep/ceiling.py."
        )
        frame = frame[columns].copy()
        # MLBAMID is the ONLY key to Savant. A null here is a player with no
        # Tier 2 forever, so count them out loud rather than discovering it as
        # a mysteriously unmeasured row later.
        n_no_id = int(frame["MLBAMID"].isna().sum())
        if n_no_id:
            print(f"    {n_no_id} {player_type}s have no MLBAMID (no Savant join)")
        frame["MLBAMID"] = pd.to_numeric(frame["MLBAMID"]).astype("Int64")
        sides.append(frame)

    snapshot = pd.concat(sides, ignore_index=True)
    assert set(snapshot["player_type"]) == {"hitter", "pitcher"}, (
        f"peak snapshot must hold both sides, got "
        f"{sorted(set(snapshot['player_type']))} — one feed came back empty."
    )
    print(f"  peak snapshot: {len(snapshot)} rows")
    return snapshot

# ── Tier 2: Savant leaderboards (public, no auth) ─────────────────────────

_SAVANT_BASE = "https://baseballsavant.mlb.com/leaderboard"

# Savant column -> our column. One entry per leaderboard.
_BAT_TRACKING_COLUMNS: dict[str, str] = {
    "id": "MLBAMID",
    "name": "savant_name",
    "avg_bat_speed": "bat_speed",
    "swing_length": "swing_length",
    "blast_per_swing": "blast_rate",
    "squared_up_per_swing": "squared_up_rate",
    "hard_swing_rate": "hard_swing_rate",
    "whiff_per_swing": "whiff_per_swing",
}
_EXIT_VELO_COLUMNS: dict[str, str] = {
    "player_id": "MLBAMID",
    "max_hit_speed": "max_ev",
    "ev50": "ev50",
    "ev95percent": "hard_hit_rate",
    "brl_percent": "barrel_rate",
}
_SPRINT_COLUMNS: dict[str, str] = {
    "player_id": "MLBAMID",
    "sprint_speed": "sprint_speed",
}
_PITCHER_COLUMNS: dict[str, str] = {
    "player_id": "MLBAMID",
    "fastball_avg_speed": "fb_velo",
    "whiff_percent": "whiff_rate",
    "avg_release_extension": "extension",
}

# Every Tier-2 tool, by player type. Percentiles are computed for all of them.
HITTER_TOOLS: tuple[str, ...] = (
    "bat_speed", "max_ev", "ev50", "barrel_rate", "hard_hit_rate",
    "blast_rate", "sprint_speed",
)
PITCHER_TOOLS: tuple[str, ...] = ("fb_velo", "whiff_rate", "extension")

# Measured, percentiled, and deliberately NOT tools. A tool gates the tail; a
# diagnostic explains HOW a tool is being produced, which is a different
# question and must not be allowed to satisfy the screen's tool bar.
#
# Squared-up rate is the case that forced this distinction. It is the most
# direct available measure of barrel-to-ball accuracy, and a hitter can post an
# elite barrel rate with a terrible one — by selling out to lift the ball rather
# than by hitting it squarely. Folding it into `pct_best` would let that hitter
# clear the tool bar on the very metric that indicts him. Kept separate, it
# contradicts the barrel rate instead, which is the finding (see
# `add_profile_flags`).
HITTER_DIAGNOSTICS: tuple[str, ...] = (
    "squared_up_rate", "whiff_per_swing", "hard_swing_rate", "swing_length",
)
PITCHER_DIAGNOSTICS: tuple[str, ...] = ()

# Metrics where a LOWER raw value is better. Their percentiles are inverted so
# that, everywhere in this module, a high percentile means good — without this,
# `pct_whiff_per_swing = 87` reads as a strength and is the opposite, and a
# reader has to remember which columns are backwards. Nothing downstream should
# ever have to know the polarity of an individual metric.
LOWER_IS_BETTER: frozenset[str] = frozenset({"whiff_per_swing"})

# The tools that constrain the TAIL rather than merely describe the player.
# This split is the whole point of the screen rule: sprint speed is a real tool
# but it caps out at ~30 SB, which wins one category, while bat speed and max EV
# are what gate a 40-homer season. A hitter whose only top-quartile tool is his
# legs has no power ceiling, and a contact bat with no impact metric has none
# either — both fail the screen on `pct_core` even though `pct_best` looks fine.
HITTER_CORE_TOOLS: tuple[str, ...] = ("bat_speed", "max_ev", "barrel_rate")
PITCHER_CORE_TOOLS: tuple[str, ...] = ("fb_velo", "whiff_rate")

# Savant's leaderboards are MLB-only, so a low-minors prospect has no row at
# all. That is a data gap, NOT a zero: it must never read as "no tools".
_MIN_SAVANT_ROWS = 100

# Playing-time floors on the Savant leaderboards. Deliberately loose — a
# 40-swing cup of coffee is a real bat-speed measurement even when it is a
# meaningless slash line, and cup-of-coffee prospects are exactly who this
# script exists to evaluate.
_MIN_SWINGS = 25
_MIN_BATTED_BALLS = 10
_MIN_PITCHES = 50


def _savant_csv(endpoint: str, params: dict, label: str) -> pd.DataFrame:
    """Fetch one Savant leaderboard as CSV.

    Savant answers a failed query with an HTML error page and HTTP 200, so a
    bare `read_csv` would raise something unreadable ten frames deep. Check the
    body looks like CSV here, where the message can name the endpoint.
    """
    response = requests.get(
        f"{_SAVANT_BASE}/{endpoint}", params={**params, "csv": "true"}, timeout=120
    )
    assert response.status_code == 200, (
        f"Savant {label} returned HTTP {response.status_code}. "
        f"URL: {response.url}. Body: {response.text[:200]}"
    )
    assert not response.text.lstrip().startswith("<"), (
        f"Savant {label} returned HTML, not CSV — the leaderboard rejected the "
        f"query. URL: {response.url}. Body: {response.text[:200]}. Savant "
        f"renames leaderboard params without notice; open that URL in a browser "
        f"and re-derive the params in _savant_csv's callers."
    )
    # utf-8-sig: Savant prefixes a BOM, which otherwise becomes part of the
    # first column name and breaks every rename below.
    frame = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
    assert len(frame) >= _MIN_SAVANT_ROWS, (
        f"Savant {label} returned only {len(frame)} rows (expected "
        f">={_MIN_SAVANT_ROWS}). A near-empty leaderboard means the season or "
        f"minimum filter is wrong, not that nobody qualified. URL: {response.url}"
    )
    print(f"  savant {label}: {len(frame)} rows")
    return frame


def _select(frame: pd.DataFrame, columns: dict[str, str], label: str) -> pd.DataFrame:
    """Rename and subset one leaderboard, failing loudly on a vanished column."""
    missing = [c for c in columns if c not in frame.columns]
    assert not missing, (
        f"Savant {label} is missing column(s) {missing}.\n"
        f"  Columns returned: {sorted(frame.columns)}\n"
        f"Savant renamed a field — update the *_COLUMNS map in data_prep/ceiling.py."
    )
    return frame[list(columns)].rename(columns=columns)


def fetch_savant_snapshot(season: int) -> pd.DataFrame:
    """Fetch every Tier-2 tool leaderboard for one season into one raw snapshot.

    Four public leaderboards, no authentication: bat tracking (bat speed, swing
    length, blast rate), exit velocity & barrels (max EV, EV50, hard-hit%,
    barrel%), sprint speed, and a pitcher custom leaderboard (fastball velocity,
    whiff%, extension).

    Args:
        season: Statcast season to pull.

    Returns:
        One row per player per type, columns: MLBAMID, savant_name,
        player_type, season, and the union of HITTER_TOOLS and PITCHER_TOOLS.
        The opposite type's tool columns are NULL, not zero — raw-layer
        discipline, same as `scrape_fangraphs.build_snapshot`.

    Note:
        Stuff+ is NOT here. It is a FanGraphs metric, not a Savant one, and the
        per-pitch Stuff+ leaderboard is a different (paywalled) endpoint; whiff
        rate plus fastball velocity plus extension is the measured-not-modelled
        substitute. CSW% is likewise absent: Savant's custom leaderboard has no
        called-strike-plus-whiff selection.
    """
    print(f"Fetching Savant tool leaderboards for {season}...")
    tracking = _select(
        _savant_csv(
            "bat-tracking",
            {"type": "batter", "minSwings": _MIN_SWINGS, "minGroupSwings": 1,
             "seasonStart": season, "seasonEnd": season},
            "bat tracking",
        ),
        _BAT_TRACKING_COLUMNS,
        "bat tracking",
    )
    exit_velo = _select(
        _savant_csv(
            "statcast",
            {"type": "batter", "year": season, "position": "", "team": "",
             "min": _MIN_BATTED_BALLS},
            "exit velocity",
        ),
        _EXIT_VELO_COLUMNS,
        "exit velocity",
    )
    sprint = _select(
        _savant_csv(
            "sprint_speed",
            {"year": season, "position": "", "team": "", "min": _MIN_BATTED_BALLS},
            "sprint speed",
        ),
        _SPRINT_COLUMNS,
        "sprint speed",
    )
    pitchers = _select(
        _savant_csv(
            "custom",
            {"year": season, "type": "pitcher", "filter": "", "min": _MIN_PITCHES,
             "selections": ",".join(
                 c for c in _PITCHER_COLUMNS if c != "player_id"
             ),
             "sort": "whiff_percent", "sortDir": "desc"},
            "pitcher stuff",
        ),
        _PITCHER_COLUMNS,
        "pitcher stuff",
    )

    hitters = tracking.merge(exit_velo, on="MLBAMID", how="outer").merge(
        sprint, on="MLBAMID", how="outer"
    )
    hitters["player_type"] = "hitter"
    pitchers["player_type"] = "pitcher"

    snapshot = pd.concat([hitters, pitchers], ignore_index=True)
    snapshot["season"] = season
    snapshot["MLBAMID"] = pd.to_numeric(snapshot["MLBAMID"]).astype("Int64")

    n_dup = int(snapshot.duplicated(["MLBAMID", "player_type"]).sum())
    assert n_dup == 0, (
        f"Savant snapshot has {n_dup} duplicate (MLBAMID, player_type) rows. "
        f"The leaderboards are one-row-per-player; a duplicate means one of "
        f"them was grouped (check that no groupBy param leaked in)."
    )
    print(
        f"  savant snapshot: {len(snapshot)} rows "
        f"({int((snapshot['player_type'] == 'hitter').sum())} hitters, "
        f"{int((snapshot['player_type'] == 'pitcher').sum())} pitchers)"
    )
    return snapshot


# ── Tier 1 scoring ────────────────────────────────────────────────────────

_SCORING_STATS: tuple[str, ...] = (
    "PA", "IP", "R", "HR", "RBI", "SB", "OPS", "W", "SV", "K", "ERA", "WHIP",
)


def _merge_role_saves(peak: pd.DataFrame, ros: pd.DataFrame) -> pd.DataFrame:
    """Replace OOPSY's all-zero SV column with a real SV projection.

    OOPSY peak sets SV (and HLD, L, QS) to exactly 0.0 for all 4,256 pitchers,
    because a save is an OPPORTUNITY outcome — a function of a manager's bullpen
    hierarchy — not a talent that can be projected from a pitcher's stuff. That
    is defensible for OOPSY and fatal here: `add_fantasy_value` z-scores SV, and
    a category with zero variance trips its std assertion.

    Filling SV with a constant would not help (still zero variance) and
    inventing one would be worse, so SV is taken from the rest-of-season system
    that DOES model bullpen roles. Mixing scales is harmless: a z-score is
    scale-free, so z(SV) reads as "how much closer to a closer's role than the
    average pitcher", which is exactly the opportunity term the league's SV
    category rewards. A pitcher with no RoS row keeps SV = 0.0 — for a
    prospect that is a fact, not a gap: he has no save role.

    W is kept from OOPSY, with a caveat worth knowing: at a fixed 198/70 IP,
    peak W is close to collinear with peak ERA, so the pitcher Tier-1 score
    leans on run prevention harder than the raw five-category count implies.
    """
    saves = ros.loc[ros["player_type"] == "pitcher", ["MLBAMID", "SV"]].copy()
    saves = saves[saves["MLBAMID"].notna()].drop_duplicates("MLBAMID")
    lookup = pd.Series(saves["SV"].to_numpy(), index=saves["MLBAMID"].to_numpy())

    is_pitcher = peak["player_type"] == "pitcher"
    assert float(peak.loc[is_pitcher, "SV"].std()) == 0.0, (
        f"_merge_role_saves: OOPSY peak SV already varies "
        f"(std={peak.loc[is_pitcher, 'SV'].std():.4f}). OOPSY has started "
        f"projecting saves — drop this borrow and use its own column."
    )
    peak.loc[is_pitcher, "SV"] = (
        peak.loc[is_pitcher, "MLBAMID"].map(lookup).fillna(0.0).astype(float)
    )
    n_with = int((peak.loc[is_pitcher, "SV"] > 0).sum())
    print(
        f"  borrowed SV from {SAVES_SYSTEM}: {n_with} of "
        f"{int(is_pitcher.sum())} pitchers carry a save role"
    )
    return peak


def add_tier1_score(peak: pd.DataFrame, ros: pd.DataFrame) -> pd.DataFrame:
    """Add 'tier1_FV': peak-talent fantasy value, defence excluded by construction.

    Runs the OOPSY peak category line through `add_fantasy_value` — the one
    place in this repo that knows how to z-score the ten league categories with
    the counting-vs-ratio distinction handled. No z-score code lives here.

    OOPSY's WAR column is dropped rather than scored: it is the only column in
    the feed that carries defensive value, and defensive value is worth zero in
    a roto league. Scoring the category line instead is what "defence-stripped"
    means operationally.

    Requires (peak): player_type, MLBAMID, and the raw FanGraphs peak columns
        PA/R/HR/RBI/SB/OPS (hitters), IP/GS/W/SV/SO/ERA/WHIP (pitchers).
    Requires (ros): player_type, MLBAMID, SV.
    Adds: K (renamed from SO), role, tier1_FV. Drops: WAR.

    Note:
        Hitters and pitchers are each z-scored over their WHOLE type
        population, starters and relievers together, so tier1_FV stays
        comparable across roles. Splitting them would make the two halves
        unrankable against each other, and the split is already priced in: a
        reliever's peak line is normalised to 70 IP against a starter's 198, so
        he correctly earns fewer strikeouts and wins.
    """
    peak = peak.copy()
    assert "SO" in peak.columns, (
        f"add_tier1_score: peak frame has no SO column, got "
        f"{sorted(peak.columns)}. The FanGraphs pitcher feed calls strikeouts "
        f"'SO'; this repo's scoring calls them 'K'."
    )
    peak = peak.rename(columns={"SO": "K"}).drop(columns=["WAR"], errors="ignore")
    peak = _merge_role_saves(peak, ros)

    # GS is a bimodal role flag in the peak feed (33 or 0), not a projection.
    peak["role"] = np.where(
        peak["player_type"] == "pitcher",
        np.where(peak.get("GS", 0) > 0, "SP", "RP"),
        "H",
    )

    # The snapshot leaves the opposite type's stats NULL (raw-layer discipline).
    # add_fantasy_value's contract is the silver-table one: every scoring stat
    # is a real number, 0 for the wrong type. Zero-filling is safe for the ratio
    # columns too — the PA/IP volume gate excludes them from the ratio z-scores,
    # which is exactly why that gate exists.
    for stat in _SCORING_STATS:
        assert stat in peak.columns, (
            f"add_tier1_score: peak frame is missing '{stat}'. Expected the "
            f"columns written by scrape_fangraphs.build_snapshot."
        )
        peak[stat] = pd.to_numeric(peak[stat], errors="coerce").fillna(0.0)

    peak = add_fantasy_value(peak)
    return peak.rename(columns={"FV": "tier1_FV"})


# ── Tier 2 scoring ────────────────────────────────────────────────────────


def add_tool_percentiles(players: pd.DataFrame) -> pd.DataFrame:
    """Add 'pct_<tool>' plus pct_best / best_tool / pct_core for Tier 2.

    Percentiles are computed HERE from raw metric values, never read from
    Savant's rendered sliders: two fetches of the same Savant player page have
    returned contradictory percentiles for the same player, so the published
    number is not reproducible. `rank(pct=True)` over the snapshot on disk is.

    Ranking is within `player_type` and over the players who actually HAVE the
    metric — a missing measurement yields NaN, never a percentile. A prospect
    who has never faced major-league pitching has no Savant row at all, and
    "unmeasured" must not read as "0th percentile", which would look identical
    to a genuinely slow bat.

    Percentiles for metrics in `LOWER_IS_BETTER` are inverted, so a high
    `pct_*` means good for EVERY metric on the frame with no exceptions to
    remember.

    Requires: player_type, plus any subset of HITTER_TOOLS / PITCHER_TOOLS.
    Adds: pct_<tool> for every tool present, n_tools, pct_best, best_tool,
        pct_core.

    Note:
        `pct_best` is a MAX, not a mean. Ceiling is set by a player's single
        loudest tool — averaging tools measures how well-rounded he is, which is
        an expected-value question, not a tail question.
    """
    players = players.copy()
    is_hitter = players["player_type"] == "hitter"

    for diagnostics, mask in (
        (HITTER_DIAGNOSTICS, is_hitter),
        (PITCHER_DIAGNOSTICS, ~is_hitter),
    ):
        for metric in [d for d in diagnostics if d in players.columns]:
            values = pd.to_numeric(players.loc[mask, metric], errors="coerce")
            ascending = metric not in LOWER_IS_BETTER
            players.loc[mask, f"pct_{metric}"] = values.rank(
                pct=True, ascending=ascending
            )

    for tools, core, mask in (
        (HITTER_TOOLS, HITTER_CORE_TOOLS, is_hitter),
        (PITCHER_TOOLS, PITCHER_CORE_TOOLS, ~is_hitter),
    ):
        present = [t for t in tools if t in players.columns]
        assert present, (
            f"add_tool_percentiles: none of {tools} are columns on the frame; "
            f"got {sorted(players.columns)}. Refresh the Savant snapshot."
        )
        for tool in present:
            values = pd.to_numeric(players.loc[mask, tool], errors="coerce")
            players.loc[mask, f"pct_{tool}"] = values.rank(pct=True)

        pct_cols = [f"pct_{t}" for t in present]
        core_cols = [f"pct_{t}" for t in core if t in present]
        block = players.loc[mask, pct_cols]
        players.loc[mask, "n_tools"] = block.notna().sum(axis=1)
        players.loc[mask, "pct_best"] = block.max(axis=1)
        players.loc[mask, "pct_core"] = players.loc[mask, core_cols].max(axis=1)

        # idxmax raises outright on an all-NA row, and most rows here ARE
        # all-NA: the peak feed covers ~7,000 players and Savant covers ~850.
        # An unmeasured player has no best tool, which is a NULL, not an error.
        measured = block.notna().any(axis=1)
        best = pd.Series(pd.NA, index=block.index, dtype="object")
        best.loc[measured] = (
            block.loc[measured].idxmax(axis=1).str.removeprefix("pct_")
        )
        players.loc[mask, "best_tool"] = best

    measured = int((players["n_tools"] > 0).sum())
    print(
        f"Tier 2: {measured} of {len(players)} players carry at least one "
        f"measured tool ({len(players) - measured} unmeasured — no MLB Statcast row)"
    )
    return players


LIFT_DEPENDENT_BARREL_PCT: float = 0.75
LIFT_DEPENDENT_SQUARED_UP_PCT: float = 0.25


def add_profile_flags(players: pd.DataFrame) -> pd.DataFrame:
    """Add 'profile_flag': internal contradictions between measured metrics.

    Two metrics that disagree is a finding, not a data-quality problem, and it
    is the finding a single ranked column cannot express. The screen answers
    "is the tail big enough"; this answers "what has to go right for it".

    `lift-dependent` is the one that matters. An elite barrel rate resting on a
    bottom-quartile squared-up rate means the power comes from swing shape
    rather than from meeting the ball, so it arrives bundled with strikeouts and
    has no short-to-it fallback when the league adjusts. Owen Caissie is the
    worked example: 94th-percentile barrels, 3rd-percentile squared-up. That
    profile still has a real 35-homer tail — it is not a rejection — but the tail
    is narrower and more fragile than the barrel rate alone implies.

    `squares-up-without-power` is the mirror image: contact accuracy with no
    impact behind it. The bat works; the ceiling is a batting-average asset.

    Requires: player_type and the pct_* columns from add_tool_percentiles.
    Adds: profile_flag (empty string where nothing contradicts or data is thin).
    """
    players = players.copy()
    flag = pd.Series("", index=players.index, dtype="object")
    barrel = players.get("pct_barrel_rate")
    squared = players.get("pct_squared_up_rate")
    if barrel is None or squared is None:
        players["profile_flag"] = flag
        return players

    both = barrel.notna() & squared.notna()
    flag = flag.mask(
        both
        & (barrel >= LIFT_DEPENDENT_BARREL_PCT)
        & (squared <= LIFT_DEPENDENT_SQUARED_UP_PCT),
        "lift-dependent: elite barrels ("
        + (barrel * 100).round(0).astype("string")
        + "th) on bottom-quartile squared-up ("
        + (squared * 100).round(0).astype("string")
        + "th) — power comes from swing shape, not contact; expect strikeouts "
        "and no fallback when the league adjusts",
    )
    flag = flag.mask(
        both
        & (squared >= LIFT_DEPENDENT_BARREL_PCT)
        & (barrel <= LIFT_DEPENDENT_SQUARED_UP_PCT),
        "squares-up-without-power: "
        + (squared * 100).round(0).astype("string")
        + "th squared-up on "
        + (barrel * 100).round(0).astype("string")
        + "th barrels — the bat works, the ceiling is average-shaped",
    )
    players["profile_flag"] = flag
    n = int((flag != "").sum())
    print(f"Profile flags: {n} player(s) carry an internal contradiction")
    return players


# ── Tier 3 ────────────────────────────────────────────────────────────────

PECOTA_TAIL_COLUMNS: tuple[str, ...] = ("MLBAMID", "p90_OPS", "p90_HR", "p90_SB")


def load_pecota_tail(path: str | None) -> pd.DataFrame | None:
    """Load a Baseball Prospectus PECOTA 90th-percentile export, if configured.

    PECOTA is the only public source that publishes an explicit per-player
    DISTRIBUTION (10th/50th/90th percentile lines), and the 90th percentile is
    literally the quantity this whole module is approximating. It is behind a BP
    subscription with no public endpoint, so there is nothing to scrape: the
    only honest hook is "point me at a CSV you exported yourself".

    Args:
        path: Path to a CSV holding PECOTA_TAIL_COLUMNS. None means not
            configured, and the caller falls back to the Tier-1 + Tier-2
            approximation.

    Returns:
        The frame, or None when `path` is None.
    """
    if path is None:
        print(
            "Tier 3: PECOTA not configured — combined ceiling score is an "
            "APPROXIMATION from Tier 1 + Tier 2, not a real 90th percentile. "
            "Pass --pecota-csv with a BP export to use the true tail."
        )
        return None

    from pathlib import Path

    csv_path = Path(path).expanduser()
    assert csv_path.is_file(), (
        f"load_pecota_tail: no file at {csv_path}. PECOTA percentiles are "
        f"paywalled and have no public endpoint: log into "
        f"baseballprospectus.com, export the PECOTA percentile projections, "
        f"and save them as a CSV with columns {list(PECOTA_TAIL_COLUMNS)}."
    )
    frame = pd.read_csv(csv_path)
    missing = [c for c in PECOTA_TAIL_COLUMNS if c not in frame.columns]
    assert not missing, (
        f"load_pecota_tail: {csv_path} is missing column(s) {missing}. "
        f"Required: {list(PECOTA_TAIL_COLUMNS)}."
    )
    print(f"Tier 3: loaded {len(frame)} PECOTA 90th-percentile rows from {csv_path}")
    return frame


# ── The screen ────────────────────────────────────────────────────────────

# How many z-units of Tier-1 score one z-unit of physical tool is worth. Tuning
# knob, not a derived constant: tier1_FV is a sum of five z-scores (range ~+-6)
# and the tool term is a single z (~+-2.5), so this sets how loudly the tail
# proxy speaks over the mean. 2.5 puts a 99th-percentile tool at roughly the
# same weight as a full point of five-category peak projection.
DEFAULT_TOOL_WEIGHT: float = 2.5

# Screen thresholds. Top quartile for "has a real tool", median for "the tool
# that actually gates the tail is not absent".
DEFAULT_TOOL_PCT_BAR: float = 0.75
DEFAULT_CORE_PCT_BAR: float = 0.50

# The "a median regular is worth zero" dial, and the number most worth tuning.
# tier1_FV is a sum of five z-scores over the whole peak population, so it is
# NOT on a 0-1 scale: measured on the 2026-08-21 snapshot the distribution runs
# median -0.36, 90th percentile 4.29, 99th 10.42, max 24.1 (Judge). 4.0 is
# therefore roughly "top decile of everyone OOPSY projects", which in a 7-team
# league with 196 roster spots is about where a peak season stops being free.
DEFAULT_FV_BAR: float = 4.0


def add_ceiling_score(
    players: pd.DataFrame,
    fv_bar: float,
    tool_pct_bar: float = DEFAULT_TOOL_PCT_BAR,
    core_pct_bar: float = DEFAULT_CORE_PCT_BAR,
    tool_weight: float = DEFAULT_TOOL_WEIGHT,
) -> pd.DataFrame:
    """Add 'ceiling_score' and the pass/fail screen, with TWO bars by horizon.

    THE STAR BAR (prospects, and anyone whose status is unclear). Unchanged from
    the original screen, and it demands BOTH halves of the thesis because either
    alone is a known false positive in a league this shallow:

      * a top-quartile physical tool (Tier 2), AND
      * a peak projection above `fv_bar` (Tier 1),

    plus one more cut that catches the two archetypes which sail through a naive
    "has a good tool" test: the speed-only player whose bat speed is below
    average, and the contact bat with no impact metric. Both fail on `pct_core` —
    the max over the tools that actually gate the tail (bat speed / max EV /
    barrel rate for hitters, fastball velocity / whiff rate for pitchers) —
    while looking fine on `pct_best`. This is the right bar for a prospect
    because his median costs four or five years of a roster slot to arrive at a
    player available free off waivers: only the tail pays that rent.

    THE MAJOR-LEAGUER BAR. He passes on max(production, breakout option):

        now_value > replacement level at his own position   (production)
          OR
        the full star bar above                             (breakout option)

    Why THAT production bar and not a second hand-tuned constant: holding a
    major leaguer costs nothing, so the only question is whether he beats the
    alternative — and the alternative is not "the average hitter", it is the
    best player at his position with no starting job. `add_positional_replacement`
    already computes exactly that number. Using it means the bar is derived from
    the league rather than invented, and it moves on its own as the league does.
    The disjunction is what the doctrine requires in both directions: a
    productive big leaguer with no upside is a legitimate HOLD (his median is
    free), and an unproductive one with a real peak is a legitimate STASH (the
    option is free too). Only a player who is both below replacement now and
    without a tail is a genuine drop.

    Otto Lopez is the case that forced this. At tier1_FV 2.60 he fails the star
    bar, and under one flat bar the board called him a reject — a prospect
    verdict on a 24-steal everyday second baseman. He now passes on production
    with a reason that says so, which is the actionable statement: not "bad
    player" but "hold with no option value, displaceable by a higher-ceiling
    alternative at 2B".

    `screen_reason` names WHICH BAR was applied and why, for passers and
    failures alike, because "rejected" is not actionable and neither is
    "accepted": "HOLD on production, no breakout option" and "star bar, peak
    projects as a regular" lead to different transactions.

    Args:
        players: Frame carrying tier1_FV, the pct_* columns, horizon,
            now_value and replacement_now.
        fv_bar: Minimum tier1_FV for the STAR bar. This is the "a median costs
            years of a roster slot" dial and applies to prospects; major
            leaguers are not held to it unless production already failed.
        tool_pct_bar: Minimum pct_best to count as having a real tool.
        core_pct_bar: Minimum pct_core, the anti-one-trick cut.
        tool_weight: z-units of ceiling per z-unit of best tool.

    Requires: player_type, tier1_FV, pct_best, pct_core, best_tool, n_tools,
        horizon, now_value, replacement_now, now_vs_replacement,
        replacement_slot.
    Adds: tool_z, ceiling_score, screen_bar, screen_pass, screen_reason.

    Note:
        With no PECOTA (Tier 3), ceiling_score is
        `tier1_FV + tool_weight * z(pct_best)` — an APPROXIMATION of the tail,
        not a measured 90th percentile. The tool percentile is mapped back
        through the normal quantile so both terms are in the same z units;
        percentiles are clipped off 0 and 1 first, because rank(pct=True)
        returns an exact 1.0 for the population maximum and ppf(1) is +inf.

        ceiling_score is deliberately NaN for a player with no Savant
        measurement, and most of the peak feed is in that state (~6,300 of
        ~7,100 rows). Substituting his tier1_FV alone would rank him as though
        the tail term were average, which is exactly the plausible-looking
        invented number this module refuses to produce. He fails the screen with
        a reason that says so, and NaN sorts to the bottom of the board.
    """
    players = players.copy()
    for col in ("tier1_FV", "pct_best", "pct_core", "best_tool", "n_tools",
                "horizon", "now_value", "replacement_now", "now_vs_replacement",
                "replacement_slot"):
        assert col in players.columns, (
            f"add_ceiling_score: missing '{col}'. Run add_tier1_score, "
            f"add_tool_percentiles, add_now_value, add_eligibility and "
            f"add_positional_replacement first."
        )

    n = max(len(players), 2)
    clipped = players["pct_best"].clip(0.5 / n, 1.0 - 0.5 / n)
    players["tool_z"] = norm.ppf(clipped)
    players["ceiling_score"] = players["tier1_FV"] + tool_weight * players["tool_z"]

    unmeasured = players["n_tools"].fillna(0) == 0
    below_fv = players["tier1_FV"] < fv_bar
    no_tool = players["pct_best"] < tool_pct_bar
    no_core = players["pct_core"].isna() | (players["pct_core"] < core_pct_bar)

    best = players["best_tool"].fillna("none")
    pct_best_str = (players["pct_best"] * 100).round(0).astype("string").fillna("n/a")
    core_kind = np.where(
        players["player_type"] == "hitter", "power", "stuff"
    )

    # ── the star bar, unchanged ───────────────────────────────────────────
    star_reason = pd.Series("", index=players.index, dtype="object")
    # Reasons are assigned worst-first so the LAST write wins and the message
    # names the most fundamental failure, not an incidental one.
    star_reason = star_reason.mask(
        no_core,
        "no "
        + pd.Series(core_kind, index=players.index)
        + " tool above the "
        + f"{core_pct_bar:.0%}"
        + " bar (pct_core="
        + (players["pct_core"] * 100).round(0).astype("string").fillna("n/a")
        + ") — one-trick profile, tail is capped",
    )
    star_reason = star_reason.mask(
        no_tool,
        "no top-quartile physical tool (best is "
        + best
        + " at "
        + pct_best_str
        + "th, bar "
        + f"{tool_pct_bar:.0%})",
    )
    star_reason = star_reason.mask(
        below_fv,
        "Tier 1 peak FV "
        + players["tier1_FV"].round(2).astype(str)
        + f" below bar {fv_bar:.2f} — peak projects as a regular, not a star",
    )
    star_reason = star_reason.mask(
        unmeasured,
        "no Savant measurement — never faced MLB pitching, so the physical "
        "ceiling is unknown (scouting reports, not this script)",
    )
    star_pass = star_reason == ""

    # ── the horizon split ─────────────────────────────────────────────────
    major = players["horizon"] == "major-leaguer"
    now_pass = players["now_value"].notna() & (players["now_vs_replacement"] > 0)
    players["screen_bar"] = np.where(major, "now|star", "star")
    players["screen_pass"] = np.where(major, star_pass | now_pass, star_pass)

    fv_str = players["tier1_FV"].round(2).astype(str)
    now_str = players["now_value"].round(2).astype("string").fillna("unmeasured")
    repl_str = players["replacement_now"].round(2).astype("string").fillna("n/a")
    slot_str = players["replacement_slot"].astype("string").fillna("no-position")
    gap_str = players["now_vs_replacement"].round(2).astype("string").fillna("n/a")
    bar_note = pd.Series(
        np.where(
            major,
            "major-leaguer bar = max(production, breakout); ",
            players["horizon"].astype(str)
            + " bar = star only, a median costs years of a roster slot; ",
        ),
        index=players.index,
    )

    reason = pd.Series("", index=players.index, dtype="object")
    reason = reason.mask(
        ~major & ~star_pass,
        bar_note + "FAILED: " + star_reason,
    )
    reason = reason.mask(
        ~major & star_pass,
        bar_note
        + "PASS: peak FV "
        + fv_str
        + " on "
        + best
        + " at "
        + pct_best_str
        + "th — the star tail is the only outcome that pays for the slot",
    )
    reason = reason.mask(
        major & ~star_pass & ~now_pass,
        bar_note
        + "FAILED BOTH: production "
        + now_str
        + " is below "
        + slot_str
        + " replacement "
        + repl_str
        + ", and no breakout option ("
        + star_reason
        + ") — nothing to hold: below the free alternative now, and no tail to "
        "take an option on",
    )
    reason = reason.mask(
        major & now_pass & ~star_pass,
        bar_note
        + "PASS on production: now_value "
        + now_str
        + " clears "
        + slot_str
        + " replacement "
        + repl_str
        + " by "
        + gap_str
        + ". NO option value ("
        + star_reason
        + ") — holding him costs nothing, so this is a legitimate hold that a "
        "higher-ceiling alternative at the same slot can displace",
    )
    reason = reason.mask(
        major & star_pass & ~now_pass,
        bar_note
        + "PASS on the breakout: peak FV "
        + fv_str
        + " on "
        + best
        + " at "
        + pct_best_str
        + "th, while production "
        + now_str
        + " sits below "
        + slot_str
        + " replacement "
        + repl_str
        + " — a stash, and the option is free because he is already here",
    )
    reason = reason.mask(
        major & star_pass & now_pass,
        bar_note
        + "PASS on both: now_value "
        + now_str
        + " above "
        + slot_str
        + " replacement "
        + repl_str
        + " AND peak FV "
        + fv_str
        + f" above {fv_bar:.2f}"
        + " on "
        + best
        + " at "
        + pct_best_str
        + "th",
    )
    players["screen_reason"] = reason

    n_major = int(major.sum())
    print(
        f"Screen: {int(players['screen_pass'].sum())} of {len(players)} pass "
        f"(fv_bar={fv_bar}, tool_pct_bar={tool_pct_bar}, core_pct_bar={core_pct_bar})"
    )
    print(
        f"  by bar: {int((major & now_pass & ~star_pass).sum())} major leaguers "
        f"held on production alone, "
        f"{int((major & star_pass).sum())} of {n_major} on a breakout option, "
        f"{int((~major & star_pass).sum())} prospects/unclear on the star bar"
    )
    return players


# ── Ownership ─────────────────────────────────────────────────────────────


def add_ownership(players: pd.DataFrame, fantrax: pd.DataFrame) -> pd.DataFrame:
    """Add 'ownership' (mine / owned / free agent / UNKNOWN) plus Position, age.

    The single most dangerous silent failure in this pipeline is a name that
    fails to join and reads as a free agent. Fantrax spells a split two-way
    player "Shohei Ohtani-P", carries its own misspellings, and disagrees with
    FanGraphs on five team abbreviations. So the match is the same CASCADE
    `build.merge_fantrax` uses — (name, team) then name alone, one-to-one — and
    an unmatched row becomes ownership "UNKNOWN", never "free agent". UNKNOWN
    means "not in this league's player universe (or the join broke)", which is a
    different and much less exciting fact than "available".

    Requires (players): Name, Team, player_type.
    Requires (fantrax): name, player_type. Optional: mlb_team, owner, Position,
        age, minors_eligible, pct_rostered, fantrax_id.
    Adds: ownership, Position, age, minors_eligible, pct_rostered, fantrax_id.
    """
    players = players.copy()
    fantrax = fantrax.copy()

    def key(names: pd.Series, types: pd.Series) -> pd.Series:
        """Normalized, suffix-stripped name, qualified by player_type.

        The type qualifier keeps a two-way player's two sides apart: without it
        Ohtani's hitter row could claim his pitcher roster spot (they are owned
        by different teams in this league), which is precisely the collision
        `match_rows(unique=True)` exists to prevent.
        """
        # fillna before map: a null name must become an unmatchable key, not
        # crash normalize_name on a float, and not match another null.
        plain = (
            names.astype("string").fillna("").map(strip_name_suffix).map(normalize_name)
        )
        return plain + "|" + types.astype(str)

    left = key(players["Name"], players["player_type"])
    right = key(fantrax["name"], fantrax["player_type"])
    idx = match_rows(
        [left + "|" + _team_key(players["Team"]), left],
        [right + "|" + _team_key(fantrax["mlb_team"]), right],
        unique=True,
    )

    for col in ("owner", "Position", "age", "minors_eligible", "pct_rostered",
                "fantrax_id"):
        assert col in fantrax.columns, (
            f"add_ownership: fantrax snapshot has no '{col}' column, got "
            f"{sorted(fantrax.columns)}. Run `uv run fetch fantrax`."
        )
        players[col] = idx.map(fantrax[col])

    matched = idx.notna()
    owner = players["owner"]
    players["ownership"] = np.select(
        [~matched, owner == MY_TEAM_NAME, owner.notna()],
        ["UNKNOWN", "mine", "owned"],
        default="free agent",
    )
    counts = players["ownership"].value_counts().to_dict()
    print(f"Ownership: matched {int(matched.sum())} of {len(players)} — {counts}")

    # The one direction that matters: a Fantrax-OWNED player who did not match.
    # He is either absent from the peak feed (fine, OOPSY does not project
    # everyone) or the join dropped him (not fine). Either way, name him — a
    # silently missing owned player is how a "free agent" board grows a player
    # somebody already has.
    unmatched_owned = fantrax[
        fantrax["owner"].notna() & ~fantrax.index.isin(idx.dropna())
    ]
    if len(unmatched_owned):
        print(
            f"  {len(unmatched_owned)} Fantrax-owned player(s) are NOT on this "
            f"board (no OOPSY peak projection, or a failed name join): "
            f"{sorted(unmatched_owned['name'].astype(str))[:8]}"
        )
    return players


# ── Time to contribution: horizon, now-value, replacement level ───────────
#
# WHAT MAKES A MEDIAN OUTCOME WORTHLESS IS THE HOLDING COST, NOT THE MEDIAN.
#
# The module header above is right about prospects and wrong about major
# leaguers, and the whole difference is who pays for the roster slot:
#
#   * A PROSPECT's median costs four or five years of a slot to arrive at a
#     player who could have been claimed off waivers for free. That median is
#     not worth zero, it is worth NEGATIVE, and only a star tail pays the rent.
#   * A MAJOR LEAGUER's median costs nothing. He produces while you hold him,
#     and holding him IS a free option on a breakout. His median is worth
#     exactly what it is worth against the alternative at his position.
#
# One flat `fv_bar` therefore cannot serve both. It is a PROSPECT bar, and
# applying it to a major leaguer states a true fact ("no star tail") as a false
# conclusion ("drop him"). Otto Lopez at tier1_FV 2.60 is the worked example: a
# 24-steal everyday second baseman is a hold with no option value, not a reject.

NUM_TEAMS: int = len(FANTRAX_TEAM_IDS)

# Every starting slot in the league, and how many of each a team starts. Read
# from config.json via optimizer.config rather than restated here, so a league
# settings change cannot leave this module scoring a roster shape nobody plays.
ALL_SLOTS: dict[str, int] = {**HITTING_SLOTS, **PITCHING_SLOTS}

# Volume floor for entering `now_value`'s z-score population. NOT a quality
# filter — a gate on whether a counting-stat line means anything.
#
# This is the number that decides whether `now_value` tells the truth. The YTD
# leaderboard carries 708 hitters with a median of ~40 PA, so z-scoring raw
# counting totals over ALL of them measures ROLE, not talent: any full-time
# regular lands two z above a population made mostly of September call-ups, and
# an empty .319 batting average with 6 homers would come out looking like an
# asset purely for showing up 533 times. Gating the population to players with
# something like a quarter-season role puts Arraez back where he belongs — a
# regular being measured against other regulars.
_NOW_MIN_PA: float = 150.0
_NOW_MIN_IP: float = 30.0

_YTD_GROUP_TO_TYPE: dict[str, str] = {"hitting": "hitter", "pitching": "pitcher"}

# YTD columns carried onto the board under a `now_` prefix. The prefix is
# load-bearing: the peak feed already owns bare PA/OPS/HR/ERA/..., and merging
# the current-season line onto those names would overwrite the very columns
# `tier1_FV` was computed from with a half-season of actual playing time.
_NOW_DISPLAY: tuple[str, ...] = (
    "PA", "IP", "avg", "OPS", "HR", "SB", "R", "RBI", "K", "ERA", "WHIP", "W", "SV",
)


def _ytd_line(ytd: pd.DataFrame) -> pd.DataFrame:
    """Reshape the StatsAPI year-to-date snapshot into the scoring contract.

    `add_fantasy_value` wants PA/IP/R/HR/RBI/SB/OPS/W/SV/K/ERA/WHIP. StatsAPI
    publishes rate stats as avg/obp/slg and counting stats under its own names,
    and — the trap — has NO `ops` column at all. OPS is obp + slg by definition;
    computing it here is not an approximation. Reading a missing column would
    have been a KeyError, but silently defaulting it to zero would have made
    every hitter's rate category vanish, which is why this is spelled out.

    ERA and WHIP are derived from ER / HA / BBA over IP, and are 0.0 for a
    zero-innings row. That fill is inert rather than a lie: `add_fantasy_value`
    gates its ratio z-scores on IP > 0, exactly so an undefined rate cannot be
    read at face value (a stored ERA of 0.00 would otherwise score as the best
    pitcher alive).

    Requires (ytd): MLBAMID, name, group, PA, IP, R, HR, RBI, SB, W, SV, SOA,
        ER, HA, BBA, avg, obp, slg.
    Returns:
        One row per (MLBAMID, player_type) with the scoring columns plus avg.
    """
    unknown = sorted(set(ytd["group"].astype(str)) - set(_YTD_GROUP_TO_TYPE))
    assert not unknown, (
        f"_ytd_line: YTD snapshot has group value(s) {unknown}; expected only "
        f"{sorted(_YTD_GROUP_TO_TYPE)}. StatsAPI renamed a stat group — update "
        f"_YTD_GROUP_TO_TYPE in data_prep/ceiling.py."
    )
    for col in ("MLBAMID", "name", "PA", "IP", "R", "HR", "RBI", "SB", "W", "SV",
                "SOA", "ER", "HA", "BBA", "avg", "obp", "slg"):
        assert col in ytd.columns, (
            f"_ytd_line: YTD snapshot has no '{col}' column, got "
            f"{sorted(ytd.columns)}. Run `uv run fetch ytd`."
        )

    def num(col: str) -> pd.Series:
        return pd.to_numeric(ytd[col], errors="coerce")

    line = pd.DataFrame(
        {
            "MLBAMID": pd.to_numeric(ytd["MLBAMID"]).astype("Int64"),
            "Name": ytd["name"].astype(str),
            "player_type": ytd["group"].map(_YTD_GROUP_TO_TYPE),
        }
    )
    for col in ("PA", "IP", "R", "HR", "RBI", "SB", "W", "SV"):
        line[col] = num(col).fillna(0.0)
    line["K"] = num("SOA").fillna(0.0)
    line["avg"] = num("avg")
    line["OPS"] = (num("obp") + num("slg")).fillna(0.0)

    # `.where(> 0)` first: dividing by a zero-innings row would emit a warning
    # and an inf, and inf is not a number this pipeline should ever carry.
    innings = line["IP"].where(line["IP"] > 0)
    line["ERA"] = (9.0 * num("ER").fillna(0.0) / innings).fillna(0.0)
    line["WHIP"] = ((num("HA").fillna(0.0) + num("BBA").fillna(0.0)) / innings).fillna(0.0)

    n_dup = int(line.duplicated(["MLBAMID", "player_type"]).sum())
    assert n_dup == 0, (
        f"_ytd_line: YTD snapshot has {n_dup} duplicate (MLBAMID, player_type) "
        f"rows. byDateRange is one row per player per group; a duplicate would "
        f"fan out the now-value merge and double-count a player."
    )
    return line


def add_now_value(players: pd.DataFrame, ytd: pd.DataFrame) -> pd.DataFrame:
    """Add 'horizon' and 'now_value': what a player is producing RIGHT NOW.

    `tier1_FV` answers "how good could he be". This answers "how good is he",
    and the two are only decision-relevant TOGETHER, because the holding cost of
    a median outcome depends entirely on whether the player is already here.

    `now_value` is the current-season category line run through
    `optimizer.player_scoring.add_fantasy_value` — the same scorer, the same five
    categories, the same counting-vs-ratio treatment as `tier1_FV`. Both are
    therefore a sum of five z-scores in z units and can be read side by side.
    They are NOT the same population: `tier1_FV` z-scores ~2,900 hitters over a
    600-PA peak line, `now_value` z-scores this year's ~370 real regulars over
    their actual volume. So a two-point gap between them is a real gap, but the
    scales are similar rather than identical, and that is stated here instead of
    being papered over with a rescaling nobody could defend.

    WHY YEAR-TO-DATE AND NOT STEAMER REST-OF-SEASON. Both were on the table and
    the RoS feed is already loaded (`SAVES_SYSTEM`). YTD wins for three reasons.
    (1) The question is what he IS doing, and YTD is measured while RoS is
    modelled — a projection re-regresses the very performance being observed, so
    a breakout would be scored back down toward its own prior. (2) By late
    August RoS is a five-week stub, so its counting totals are dominated by
    games remaining; a z-score over it partly measures schedule position. (3)
    Only YTD can define the horizon split at all: RoS assigns projected PA to
    players with zero major-league experience, which would hand a Double-A
    outfielder a `now_value` and silently erase the distinction this function
    exists to draw.

    JOIN ON MLBAMID, NEVER ON NAME. Both feeds carry MLB's own id and the merge
    uses it. The failure mode is not hypothetical: an ad-hoc name join spliced
    the 35-year-old Max Muncy's 25-homer season (id 571970, LAD) onto the
    23-year-old Max Muncy's roster row (id 691777, ATH). Two real players, one
    spelling, and the result was a completely fictional recommendation. The
    row-count assert below is the tripwire.

    `horizon` is deliberately three-valued, because two would require a guess:
        major-leaguer  — has current-season MLB PA or IP. Holding him costs
                         nothing; he is producing while you decide.
        prospect       — no current-season MLB volume, and minors-eligible.
                         Holding him costs years of a roster slot.
        unclear        — no current-season MLB volume and NOT minors-eligible:
                         a veteran who missed the whole year, a player who never
                         joined to Fantrax, or a name the ownership cascade
                         dropped. He is treated as a prospect by the screen
                         (only a star tail can justify the slot, since there is
                         no production to weigh) but he is LABELLED differently,
                         so "we do not know what this is" never reads as "we
                         checked and he is a prospect".

    Requires (players): MLBAMID, player_type, minors_eligible.
    Requires (ytd): the raw `ytd` snapshot from data_prep.statsapi_stats.
    Adds: now_<stat> for each of _NOW_DISPLAY, now_value, horizon.
    """
    players = players.copy()
    for col in ("MLBAMID", "player_type", "minors_eligible"):
        assert col in players.columns, (
            f"add_now_value: missing '{col}'. Run add_tier1_score and "
            f"add_ownership before add_now_value."
        )

    line = _ytd_line(ytd)

    # Score over REGULARS only — see _NOW_MIN_PA. Everyone else keeps a NULL
    # now_value, which is the honest answer ("too little major-league volume to
    # measure") and is handled explicitly by the screen rather than being
    # imputed to zero, which would read as "average" and is a number nobody
    # earned.
    enough = (
        ((line["player_type"] == "hitter") & (line["PA"] >= _NOW_MIN_PA))
        | ((line["player_type"] == "pitcher") & (line["IP"] >= _NOW_MIN_IP))
    )
    # No "is the snapshot thick enough" assert here on purpose: an early-season
    # YTD file is caught with a far better message by
    # `add_positional_replacement`, which can name the SLOT that ran out of
    # players. A second, vaguer copy of that check would only fire first.
    scored = add_fantasy_value(line.loc[enough].reset_index(drop=True))

    carried = line[["MLBAMID", "player_type", *_NOW_DISPLAY]].rename(
        columns={c: f"now_{c}" for c in _NOW_DISPLAY}
    )
    n_before = len(players)
    players = players.merge(carried, on=["MLBAMID", "player_type"], how="left")
    players = players.merge(
        scored[["MLBAMID", "player_type", "FV"]].rename(columns={"FV": "now_value"}),
        on=["MLBAMID", "player_type"],
        how="left",
    )
    assert len(players) == n_before, (
        f"add_now_value: the YTD merge changed the row count {n_before} -> "
        f"{len(players)}. The merge key is (MLBAMID, player_type) and must be "
        f"unique on both sides; a fan-out here is the same class of bug as "
        f"joining on name (see this function's docstring on Max Muncy)."
    )

    # Volume is a COUNTING fact: no YTD row means zero major-league PA this
    # year, not an unknown. The rates stay NULL — a player with no plate
    # appearances has no OPS, and 0.000 would be a fabricated slash line.
    for col in ("now_PA", "now_IP"):
        players[col] = players[col].fillna(0.0)

    has_mlb = (players["now_PA"] > 0) | (players["now_IP"] > 0)
    minors = players["minors_eligible"] == True  # noqa: E712
    players["horizon"] = np.select(
        [has_mlb, minors], ["major-leaguer", "prospect"], default="unclear"
    )
    counts = pd.Series(players["horizon"]).value_counts().to_dict()
    print(
        f"Horizon: {counts} — now_value measured for "
        f"{int(players['now_value'].notna().sum())} players with a regular's "
        f"volume (PA>={_NOW_MIN_PA:.0f} / IP>={_NOW_MIN_IP:.0f})"
    )
    return players


def add_eligibility(players: pd.DataFrame) -> pd.DataFrame:
    """Add 'eligible_slots': every lineup slot a player can actually fill.

    Fantrax states eligibility as a comma-separated Position string ("2B,SS"),
    and the player is eligible at EVERY slot in it. Reading only the first is
    how a board goes position-blind and cheerfully recommends dropping a
    manager's only second baseman.

    Parsing is delegated to `optimizer.players.get_eligible_slots`, which is
    already the single place that knows config.json's slot_eligibility map (so
    "2B" also earns UTIL, and "UT" earns UTIL and nothing else). Re-splitting
    the string here would be a second, drifting copy of the league's rules.

    Requires: Position (added by add_ownership).
    Adds: eligible_slots, an EMPTY frozenset for a player who did not join to
        Fantrax. Empty means "eligibility unknown", and every position filter
        and replacement-level population excludes him — which is the point: a
        guessed slot is how an unmatched row acquires a roster spot.
    """
    players = players.copy()
    assert "Position" in players.columns, (
        "add_eligibility: no 'Position' column. Run add_ownership first — "
        "Position comes from the Fantrax snapshot, not from the peak feed."
    )
    slots = pd.Series(
        [frozenset()] * len(players), index=players.index, dtype="object"
    )
    known = players["Position"].notna()
    slots.loc[known] = players.loc[known, "Position"].astype(str).map(
        lambda position: frozenset(get_eligible_slots(position))
    )
    players["eligible_slots"] = slots
    n_multi = int(slots.map(len).gt(2).sum())
    print(
        f"Eligibility: {int(known.sum())} of {len(players)} players carry a "
        f"Fantrax position ({n_multi} are multi-position beyond their UTIL slot)"
    )
    return players


def add_positional_replacement(
    players: pd.DataFrame, num_teams: int = NUM_TEAMS
) -> pd.DataFrame:
    """Add the replacement level at each player's OWN position.

    `tier1_FV` and `now_value` rank a player against every hitter alive, which
    is the wrong comparison for a roster decision. A team does not choose
    between a second baseman and the field; it chooses between this second
    baseman and the next-best second baseman it could actually start. In a
    7-team league with S starting slots at a position, the 7*S players filling
    those slots are the ones with jobs, so the (7*S + 1)-th best is the first
    player who does NOT have one — the free alternative. That is replacement
    level, and value above it is the only figure a drop decision can use.

    The population is every player who is IN this league's universe
    (`ownership != "UNKNOWN"`) and has a measured `now_value`: rostered players
    count, because they are the ones occupying the slots, and free agents count,
    because they are what you would actually sign. Excluding either would move
    the bar for no reason.

    A multi-position player is counted at EVERY slot he qualifies for, and his
    own `replacement_now` is the MINIMUM over those slots. That is deliberate:
    flexibility is worth something, and the min picks the scarcest position he
    can fill, which is where he generates the most surplus and therefore where
    he would actually be deployed. `replacement_slot` names it, so the number is
    never anonymous.

    Args:
        players: Frame carrying ownership, now_value and eligible_slots.
        num_teams: Teams in the league. Defaults to the configured 7.

    Requires: ownership, now_value, eligible_slots.
    Adds: replacement_now, replacement_slot, now_vs_replacement. A player at a
        slot nobody on the frame can fill, or with no measured now_value, gets
        NULL rather than a level derived from an empty population.

    Note:
        UTIL has S = 1, so its replacement level is the 8th-best hitter in the
        league — far above any real position's. It never wins the MIN and never
        binds, which is correct rather than a bug: UTIL is where a surplus bat
        goes, not a position anyone is scarce at.
    """
    players = players.copy()
    for col in ("ownership", "now_value", "eligible_slots"):
        assert col in players.columns, (
            f"add_positional_replacement: missing '{col}'. Run add_ownership, "
            f"add_now_value and add_eligibility first."
        )

    pool = (players["ownership"] != "UNKNOWN") & players["now_value"].notna()
    levels: dict[str, float] = {}
    empty: list[str] = []
    for slot, n_slots in ALL_SLOTS.items():
        at_slot = pool & players["eligible_slots"].map(lambda s: slot in s)
        values = players.loc[at_slot, "now_value"].sort_values(ascending=False)
        index = num_teams * n_slots
        if len(values) == 0:
            # Nobody on this frame can play the slot, so it HAS no replacement
            # level. `pick` below already handles a missing slot by ignoring it,
            # and the alternative — inventing a level from an empty population —
            # is the one thing this module never does. Printed, not silent.
            empty.append(slot)
            continue
        assert len(values) > index, (
            f"add_positional_replacement: only {len(values)} measured players "
            f"are eligible at {slot}, but the league starts "
            f"{num_teams} x {n_slots} = {index} of them, so there is no "
            f"(#{index + 1}) replacement player to name. Either the Fantrax "
            f"position join is broken or the YTD snapshot is too thin."
        )
        levels[slot] = float(values.iloc[index])

    assert levels, (
        f"add_positional_replacement: not one of {sorted(ALL_SLOTS)} has an "
        f"eligible, measured player, so no replacement level exists at all. The "
        f"Fantrax position join produced nothing — check add_eligibility's "
        f"printed match count and add_now_value's coverage."
    )

    def pick(slots: frozenset) -> tuple[float, object]:
        """Scarcest slot this player can fill, i.e. the lowest bar he clears."""
        usable = [s for s in slots if s in levels]
        if not usable:
            return (np.nan, pd.NA)
        slot = min(usable, key=lambda s: levels[s])
        return (levels[slot], slot)

    picked = list(players["eligible_slots"].map(pick))
    players["replacement_now"] = [p[0] for p in picked]
    players["replacement_slot"] = [p[1] for p in picked]
    players["now_vs_replacement"] = players["now_value"] - players["replacement_now"]

    print(
        f"Replacement level (now_value of the #{{{num_teams}S+1}} player, "
        f"{num_teams} teams): "
        + ", ".join(
            f"{slot}(S={ALL_SLOTS[slot]})={level:+.2f}"
            for slot, level in levels.items()
        )
        + (f" — no eligible measured player at {empty}" if empty else "")
    )
    return players


# ── Change 3: surface the trade-off instead of resolving it ────────────────
#
# A single ranked column has to price "how much present production is one point
# of peak worth to me", and that price depends entirely on how close the manager
# is to competing — which this script cannot know. So it must not pick one.
# `compare_at_position` emits BOTH deltas side by side and refuses to combine
# them; `print_contention_context` reports the standings as CONTEXT the manager
# reads, never as a weight that silently moves a score.

# Headline rate and peak column per side, with the direction that counts as
# better. The rate is what the manager feels this week; the peak column is the
# thing he would be buying. Deliberately NOT z-scores: "costs 0.05 OPS" is a
# sentence a manager can price, "costs 0.31 z" is not.
_TRADEOFF_AXES: dict[str, tuple[str, str, str, str, float]] = {
    "hitter": ("now_OPS", "OPS", "wRC+", "peak wRC+", 1.0),
    "pitcher": ("now_ERA", "ERA", "ERA", "peak ERA", -1.0),
}

_COMPARE_COLUMNS: list[str] = [
    "Name", "Position", "age", "ownership", "horizon", "screen_bar",
    "screen_pass", "now_PA", "now_IP", "now_value", "tier1_FV", "trade_off",
]


def compare_at_position(
    players: pd.DataFrame,
    incumbent: str,
    slot: str,
    candidate_ownership: tuple[str, ...] = ("free agent",),
    top_n: int = 5,
) -> pd.DataFrame:
    """One incumbent against the alternatives at his slot, both deltas explicit.

    The output row for each candidate says, in the units the categories are
    actually scored in, what the swap COSTS in present production and what it
    BUYS in peak talent — "costs 0.048 OPS of current production, buys +22.2
    points of peak wRC+". Two numbers, not one, on purpose: collapsing them into
    a single score is exactly the judgement call (how close am I to competing?)
    that belongs to the manager and that no snapshot on disk contains.

    Candidates default to free agents because those are the swaps that need no
    counterparty. Pass `candidate_ownership=("free agent", "owned")` to scout
    trade targets, or add "mine" to compare two of your own players.

    Args:
        players: A scored ceiling table (post-`build_ceiling_table`).
        incumbent: Display name of the player currently holding the slot.
        slot: Lineup slot to compare within, e.g. "2B". Must be a configured
            slot, so a typo cannot silently return an empty board.
        candidate_ownership: Which ownership states may appear as candidates.
        top_n: How many candidates to return, best peak talent first.

    Returns:
        A frame whose FIRST row is the incumbent (deltas zero) followed by up to
        `top_n` candidates sorted by tier1_FV descending, carrying
        _COMPARE_COLUMNS, the two raw levels, their two deltas, and `pareto`.

    Note:
        Sorted by tier1_FV, i.e. by what you are BUYING, not by any blend of the
        two deltas. There is no blend in this function anywhere; that is the
        whole point of it. The one comparative claim it does make is `pareto` —
        "no other candidate beats this one on both axes at once" — which is the
        strongest statement available without an exchange rate the manager has
        not supplied.
    """
    assert slot in ALL_SLOTS, (
        f"compare_at_position: '{slot}' is not a configured lineup slot. "
        f"Available: {sorted(ALL_SLOTS)} (from config.json league.hitting_slots "
        f"and league.pitching_slots)."
    )
    for col in ("eligible_slots", "now_value", "tier1_FV", "horizon"):
        assert col in players.columns, (
            f"compare_at_position: missing '{col}'. Pass a frame from "
            f"build_ceiling_table, not a raw peak snapshot."
        )

    at_slot = players[players["eligible_slots"].map(lambda s: slot in s)]
    held = at_slot[at_slot["Name"] == incumbent]
    assert len(held) == 1, (
        f"compare_at_position: found {len(held)} rows for incumbent "
        f"{incumbent!r} eligible at {slot}. "
        + (
            f"Names eligible at {slot} sharing a surname: "
            f"{sorted(n for n in at_slot['Name'].astype(str) if n.rsplit(' ', 1)[-1] in incumbent)[:8]}"
            if len(held) == 0
            else "A two-way player has one row per side — pass the frame already "
            "filtered to one player_type."
        )
    )
    row = held.iloc[0]
    side = str(row["player_type"])
    rate_col, rate_label, peak_col, peak_label, better = _TRADEOFF_AXES[side]
    assert peak_col in players.columns, (
        f"compare_at_position: the frame has no '{peak_col}' column, so the "
        f"peak half of the trade-off cannot be stated. Keep {peak_col} in the "
        f"_PEAK_* column lists."
    )

    candidates = at_slot[
        (at_slot["Name"] != incumbent)
        & (at_slot["player_type"] == side)
        & at_slot["ownership"].isin(candidate_ownership)
    ].sort_values("tier1_FV", ascending=False).head(top_n)

    board = pd.concat([held, candidates], ignore_index=True)
    board[f"d_{rate_label}"] = board[rate_col] - row[rate_col]
    board[f"d_{peak_label}"] = board[peak_col] - row[peak_col]
    board["d_now_value"] = board["now_value"] - row["now_value"]
    board["d_tier1_FV"] = board["tier1_FV"] - row["tier1_FV"]

    def phrase(delta: float, unit: str, digits: int, verb_up: str, verb_down: str) -> str:
        """One signed delta as a sentence, with `better` folding in polarity."""
        if pd.isna(delta):
            return f"{unit} unmeasured"
        signed = delta * better
        verb = verb_up if signed > 0 else verb_down
        return f"{verb} {abs(delta):.{digits}f} {unit}"

    # Verbs, not signs. On a lower-is-better axis "gains 1.6 ERA" reads as the
    # opposite of what it means, and a trade-off nobody can parse is the same as
    # no trade-off at all.
    rate_verbs = ("improves", "worsens") if better < 0 else ("gains", "costs")
    peak_verbs = ("improves", "worsens") if better < 0 else ("buys", "gives up")

    trade_off = []
    for _, cand in board.iterrows():
        if cand["Name"] == incumbent:
            trade_off.append(f"(incumbent at {slot})")
            continue
        trade_off.append(
            phrase(cand[f"d_{rate_label}"], rate_label, 3, *rate_verbs)
            + " of current production, "
            + phrase(cand[f"d_{peak_label}"], peak_label, 1, *peak_verbs)
        )
    board["trade_off"] = trade_off

    # WHICH alternative is strongest, answered WITHOUT an exchange rate. A
    # candidate is off the frontier when some other row on the board beats it on
    # BOTH axes at once — that verdict needs no weight and no blend, which is
    # why it is the only ranking this function is willing to assert. A row with
    # an unmeasured axis stays ON the frontier: unknown is not dominated.
    rate = board[rate_col] * better
    peak = board[peak_col] * better
    board["pareto"] = [
        not bool(((rate > r) & (peak > q)).any())
        for r, q in zip(rate, peak)
    ]

    columns = _COMPARE_COLUMNS[:-1] + [
        rate_col, f"d_{rate_label}", peak_col, f"d_{peak_label}",
        "d_now_value", "d_tier1_FV", "pareto", "trade_off",
    ]
    return board[[c for c in columns if c in board.columns]]


def print_contention_context() -> None:
    """Report how far this team is from competing — as CONTEXT, not as a weight.

    Nothing in this module reads the return value, because there isn't one. That
    is deliberate: the standings are the single most tempting thing to fold into
    a score ("he's 5th, so weight present production 0.4"), and doing that would
    silently price the manager's own strategic call for him. Printed beside the
    board, the same fact lets him price it himself.

    Two teams below this one are VOLUME anomalies rather than competitors — one
    is ~1,100 AB short of a full slate and one is ~3,400 short — so the printed
    AB/IP columns are the point of the table, not decoration: a roto rank that
    includes a team which never fielded a roster is not a rank.

    Requires: a `standings` snapshot on disk (`uv run fetch standings`).
    """
    standings, date = read_latest_raw("standings")
    for col in ("team_id", "total_points", "overall_rank", "ab", "ip"):
        assert col in standings.columns, (
            f"print_contention_context: standings snapshot has no '{col}', got "
            f"{sorted(standings.columns)}. Run `uv run fetch standings`."
        )
    # Join by team_id, not team_name: Fantrax display names in the standings
    # feed drift from the roster/owner names in config.json (one team is
    # "Future Savannah Banana, Evan Carter" here and "Future AL MVP, Evan
    # Carter" there), so a name join would silently fail to find my own team.
    standings = standings.copy()
    standings["team"] = standings["team_id"].map(TEAM_ID_TO_NAME)
    my_id = FANTRAX_TEAM_IDS[MY_TEAM_NAME]
    mine = standings[standings["team_id"] == my_id]
    assert len(mine) == 1, (
        f"print_contention_context: found {len(mine)} standings rows for "
        f"{MY_TEAM_NAME} (team_id {my_id}). config.json's fantrax_team_ids and "
        f"the standings feed disagree."
    )
    mine = mine.iloc[0]
    leader = standings.sort_values("total_points", ascending=False).iloc[0]

    print(f"\nCONTENTION CONTEXT (standings snapshot {date}) — reported, NOT scored")
    print(
        f"  {MY_TEAM_NAME}: {mine['total_points']} roto points, rank "
        f"{int(mine['overall_rank'])} of {len(standings)}; leader "
        f"{leader['team']} at {leader['total_points']} "
        f"({leader['total_points'] - mine['total_points']:+.1f} ahead)"
    )
    with pd.option_context("display.width", 200):
        print(
            standings.sort_values("total_points", ascending=False)[
                ["team", "total_points", "ab", "ip"]
            ].to_string(index=False)
        )
    thin = standings[standings["ab"] < 0.8 * standings["ab"].max()]
    if len(thin):
        print(
            f"  {len(thin)} team(s) are volume anomalies rather than "
            f"competitors ({', '.join(thin['team'].astype(str))}) — they have "
            f"not fielded a full slate, so their rank below this team is not "
            f"evidence of anything. Price the contention gap yourself; this "
            f"module will not do it for you."
        )


# ── Orchestration ─────────────────────────────────────────────────────────

OUTPUT_COLUMNS: list[str] = [
    "Name", "Position", "age", "role", "ownership", "horizon",
    "now_value", "replacement_slot", "now_vs_replacement",
    "tier1_FV", "best_tool", "pct_best", "pct_core",
    "ceiling_score", "screen_bar", "screen_pass", "screen_reason", "profile_flag",
]


def build_ceiling_table(
    fv_bar: float,
    season: int | None = None,
    refresh: bool = False,
    pecota_csv: str | None = None,
    tool_pct_bar: float = DEFAULT_TOOL_PCT_BAR,
    core_pct_bar: float = DEFAULT_CORE_PCT_BAR,
    tool_weight: float = DEFAULT_TOOL_WEIGHT,
    position: str | None = None,
) -> pd.DataFrame:
    """Join all three tiers plus current production, and score every player.

    Args:
        fv_bar: Tier-1 bar for the STAR half of the screen.
        season: Statcast season for Tier 2. Defaults to the current year.
        refresh: Re-fetch the peak projections and Savant leaderboards even if
            today's snapshots already exist.
        pecota_csv: Optional Tier-3 export (see `load_pecota_tail`).
        tool_pct_bar, core_pct_bar, tool_weight: passed to `add_ceiling_score`.
        position: Keep only players eligible at this lineup slot (e.g. "2B").
            Applied LAST, after scoring: replacement level is a property of the
            whole league, so filtering first would compute the bar from a
            handful of survivors instead of from everyone at the position.

    Returns:
        One row per player-side, sorted by ceiling_score descending, carrying
        OUTPUT_COLUMNS plus every pct_* tool column, every now_* current-season
        column, and eligible_slots.
    """
    if season is None:
        season = datetime.date.today().year
    today = datetime.date.today()

    if refresh or today not in available_dates(f"projections/{PEAK_SYSTEM}"):
        write_raw(fetch_peak_snapshot(), f"projections/{PEAK_SYSTEM}")
    if refresh or today not in available_dates("savant", season=season):
        write_raw(fetch_savant_snapshot(season), "savant", season=season)

    peak, peak_date = read_latest_raw(f"projections/{PEAK_SYSTEM}")
    # savant and ytd are season-partitioned (raw_io.SEASONAL_SOURCES): both
    # describe one specific season, so reading them without one could silently
    # return a different year's data.
    savant, savant_date = read_latest_raw("savant", season=season)
    ros, ros_date = read_latest_raw(f"projections/{SAVES_SYSTEM}")
    fantrax, fantrax_date = read_latest_raw("fantrax")
    ytd, ytd_date = read_latest_raw("ytd", season=season)
    print(
        f"Snapshots: peak={peak_date} savant={savant_date} "
        f"{SAVES_SYSTEM}={ros_date} fantrax={fantrax_date} ytd={ytd_date}"
    )

    players = add_tier1_score(peak, ros)

    # MLBAMID is MLB's own id and needs no name fallback — it is the one key
    # FanGraphs and Savant genuinely share.
    # Diagnostics ride along with the tools. They are excluded from pct_best by
    # `add_tool_percentiles`, not by being dropped here — omitting them from the
    # merge silently starved `add_profile_flags` of the only two columns it
    # reads, and it degraded to a no-op without complaining.
    measured = [
        m
        for m in (*HITTER_TOOLS, *PITCHER_TOOLS,
                  *HITTER_DIAGNOSTICS, *PITCHER_DIAGNOSTICS)
        if m in savant.columns
    ]
    players = players.merge(
        savant[["MLBAMID", "player_type", *measured]],
        on=["MLBAMID", "player_type"],
        how="left",
    )
    players = add_tool_percentiles(players)
    players = add_profile_flags(players)
    players = add_ownership(players, fantrax)

    pecota = load_pecota_tail(pecota_csv)
    if pecota is not None:
        players = players.merge(pecota, on="MLBAMID", how="left")

    # Order matters: now_value needs minors_eligible from add_ownership,
    # eligibility needs Position from it, and replacement level needs both
    # now_value and eligible_slots. add_ceiling_score needs all three.
    players = add_now_value(players, ytd)
    players = add_eligibility(players)
    players = add_positional_replacement(players)
    players = add_ceiling_score(
        players, fv_bar, tool_pct_bar, core_pct_bar, tool_weight
    )
    players = players.sort_values("ceiling_score", ascending=False, ignore_index=True)

    if position is not None:
        assert position in ALL_SLOTS, (
            f"build_ceiling_table: '{position}' is not a configured lineup "
            f"slot. Available: {sorted(ALL_SLOTS)}."
        )
        keep = players["eligible_slots"].map(lambda s: position in s)
        print(
            f"Filtered to {position}-eligible: {int(keep.sum())} of "
            f"{len(players)} rows"
        )
        players = players[keep].reset_index(drop=True)
    return players


def main() -> None:
    """Entry point: build the ceiling table and print the top of it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=30, help="rows to print")
    parser.add_argument(
        "--fv-bar", type=float, default=DEFAULT_FV_BAR,
        help=f"Tier-1 peak FV bar for the screen, default {DEFAULT_FV_BAR} "
             f"(higher = only stars)",
    )
    parser.add_argument("--season", type=int, default=None, help="Statcast season")
    parser.add_argument("--refresh", action="store_true", help="re-fetch sources")
    parser.add_argument("--free-agents-only", action="store_true")
    parser.add_argument("--minors-only", action="store_true")
    parser.add_argument(
        "--position", default=None,
        help="keep only players eligible at this lineup slot, e.g. 2B or SP",
    )
    parser.add_argument(
        "--horizon", default=None,
        choices=("major-leaguer", "prospect", "unclear"),
        help="keep only one time-to-contribution class",
    )
    parser.add_argument(
        "--compare-to", default=None, metavar="NAME",
        help="print the side-by-side trade-off against this incumbent; "
             "requires --position",
    )
    parser.add_argument("--failures", action="store_true",
                        help="show rejected players and their reason instead")
    parser.add_argument("--pecota-csv", default=None, help="Tier-3 BP export")
    args = parser.parse_args()

    players = build_ceiling_table(
        fv_bar=args.fv_bar,
        season=args.season,
        refresh=args.refresh,
        pecota_csv=args.pecota_csv,
        position=args.position,
    )

    print_contention_context()

    if args.compare_to is not None:
        assert args.position is not None, (
            "--compare-to needs --position: a trade-off is only meaningful "
            "against the alternatives at the SAME slot, and 'the best available "
            "hitter' is not one of them."
        )
        board = compare_at_position(players, args.compare_to, args.position)
        print(f"\n{'=' * 100}")
        print(
            f"{args.position}: {args.compare_to} vs the free-agent "
            f"alternatives — two deltas, deliberately not blended"
        )
        print(f"{'=' * 100}")
        with pd.option_context("display.width", 260, "display.max_colwidth", 70):
            print(board.to_string(index=False))

    if args.horizon is not None:
        players = players[players["horizon"] == args.horizon]
        print(f"Filtered to horizon={args.horizon}: {len(players)} rows")
    if args.free_agents_only:
        players = players[players["ownership"] == "free agent"]
        print(f"Filtered to free agents: {len(players)} rows")
    if args.minors_only:
        players = players[players["minors_eligible"] == True]  # noqa: E712
        print(f"Filtered to minors-eligible: {len(players)} rows")

    # `screen_reason` now names the bar for passers too, but it is a sentence
    # and does not fit beside 17 other columns. The board carries `screen_bar`
    # instead; the full reason earns its width in the failures view.
    if args.failures:
        shown = players[~players["screen_pass"]]
        columns = OUTPUT_COLUMNS
        label = "REJECTED"
    else:
        shown = players[players["screen_pass"]]
        columns = [c for c in OUTPUT_COLUMNS if c != "screen_reason"]
        label = "CEILING BOARD"

    print(f"\n{'=' * 100}")
    print(f"{label} — top {args.top} of {len(shown)} (tail approximation)")
    print(f"{'=' * 100}")
    with pd.option_context("display.width", 240, "display.max_colwidth", 70):
        print(shown[columns].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
