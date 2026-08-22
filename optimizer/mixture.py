"""
Outcome mixtures: turn a player into a probability-weighted set of futures.

One abstraction covers both populations, which is the point. A major leaguer is
a mixture with a single certain branch; a prospect is a mixture over outcome
tiers and arrival years. Everything downstream (`value.branch_payoffs`) reads
only the mixture, so the three-way "is this a prospect or not" classification
that used to gate the old scoring path is gone.

WHERE THE NUMBERS COME FROM
---------------------------
`data/priors/milb_cohort.parquet` — every minor-league season 2005-2018 joined
to what that player did in MLB over the following eight years, tiered in OUR
roto categories (see data_prep.prospect_outcomes). Two things are derived here:

  the joint P(tier, arrival | state)   which future, and when it starts paying
  the tier archetype line              what that future actually looks like

JOINT, NOT A PRODUCT OF MARGINALS. Tier and arrival year are strongly dependent
— stars arrive younger and faster — so multiplying a tier marginal by an arrival
marginal would hand slow-arriving players star probabilities they never had. The
joint is read straight off the cohort.

BACKOFF IS EXPLICIT AND REPORTED. The fully-conditioned cell (age, level,
age-relative-to-level, performance) is often thin, so a cell below MIN_CELL_N
falls back to (age, level) and says so in the returned `conditioning` field. It
never silently smooths, and it never invents a cell that has no support at all.
"""

import numpy as np
import pandas as pd

from .value import annualize

# Below this many cohort observations a fully-conditioned cell is not usable and
# we back off to (age, level). Matches data_prep.prospect_outcomes.MIN_CELL_N.
MIN_CELL_N: int = 20

# Tiers, worst to best. "never" carries the whole non-arrival mass and must be
# present in every mixture or the probabilities silently under-sum.
TIERS: tuple[str, ...] = ("never", "fringe", "regular", "star")

# Scoring stats a branch line must carry.
LINE_STATS: tuple[str, ...] = (
    "PA", "IP", "R", "HR", "RBI", "SB", "OPS", "W", "SV", "K", "ERA", "WHIP",
)
_COUNTS: tuple[str, ...] = ("R", "HR", "RBI", "SB", "W", "SV", "K")
_RATES: tuple[str, ...] = ("OPS", "ERA", "WHIP")

# Full-season reference volumes the archetype lines are scaled to.
FULL_PA: float = 600.0
FULL_IP: float = 180.0

# An arrival further out than this is treated as never arriving. Not a horizon
# on VALUE -- the objective has none -- but a statement about the cohort: the
# outcome window is eight years, so an arrival at year 8 is the last one
# actually observed and anything beyond it is unmeasured, not zero-probability.
MAX_ARRIVAL: int = 8


def tier_archetype_lines(cohort: pd.DataFrame) -> pd.DataFrame:
    """The average full MLB season a member of each tier actually produced.

    Career totals over the eight-year window divided by MLB seasons played, then
    scaled to a full season's volume. Dividing by seasons played rather than by
    the window is deliberate: we want what a season LOOKS LIKE when he plays, and
    `survival` in `value.project_line` handles whether he plays at all. Baking
    absence into the line as well would apply that discount twice.

    Requires columns: player_type, tier, n_mlb_seasons, car_PA/car_IP and the
        car_* category totals.
    Returns:
        One row per (player_type, tier) with the LINE_STATS columns, plus
        n_players. The "never" tier is an all-zero line.
    """
    rows = []
    players = cohort.drop_duplicates("player_id")
    for player_type in ("hitter", "pitcher"):
        for tier in TIERS:
            block = players[
                (players["player_type"] == player_type) & (players["tier"] == tier)
            ]
            line = {stat: 0.0 for stat in LINE_STATS}
            if tier == "never" or block.empty:
                rows.append(
                    {"player_type": player_type, "tier": tier,
                     "n_players": len(block), **line}
                )
                continue

            played = block[block["n_mlb_seasons"].fillna(0) > 0]
            seasons = played["n_mlb_seasons"].astype(float)
            volume_key = "car_PA" if player_type == "hitter" else "car_IP"
            per_season_volume = (played[volume_key].astype(float) / seasons).mean()
            assert per_season_volume > 0, (
                f"tier_archetype_lines: tier {tier!r} {player_type}s have zero "
                f"{volume_key} per season. The cohort's career totals are "
                f"missing; check build_cohort populated car_*."
            )
            reference = FULL_PA if player_type == "hitter" else FULL_IP
            scale = reference / per_season_volume

            if player_type == "hitter":
                line["PA"] = reference
            else:
                line["IP"] = reference
            for category in _COUNTS:
                column = f"car_{category}"
                if column not in played.columns:
                    continue
                per_season = (played[column].astype(float) / seasons).mean()
                line[category] = float(per_season * scale)
            for category in _RATES:
                column = f"car_{category}"
                if column in played.columns:
                    value = played[column].astype(float).replace(0.0, np.nan).mean()
                    line[category] = float(value) if np.isfinite(value) else 0.0
            rows.append(
                {"player_type": player_type, "tier": tier,
                 "n_players": len(played), **line}
            )
    return pd.DataFrame(rows)


def joint_outcome_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """P(tier, arrival year | state), at two conditioning levels.

    Emits both the fully-conditioned cell and the (age, level) backoff, tagged
    in `conditioning`, so `prospect_branches` can choose per player and report
    which one it used.

    Returns:
        player_type, age, sport_id, age_rel_bucket, perf_bucket, tier,
        arrive, p, cell_n, conditioning.
        `arrive` is years from the observation season to the MLB debut, or -1
        for the never-arrives mass. Probabilities sum to 1 within each
        (conditioning, state) group.
    """
    frame = cohort.copy()
    assert "years_to_mlb" in frame.columns and "tier" in frame.columns, (
        "joint_outcome_table: cohort must carry years_to_mlb and tier. Build it "
        "with data_prep.prospect_outcomes.build_cohort."
    )
    arrive = pd.to_numeric(frame["years_to_mlb"], errors="coerce")
    # Never arrived, or arrived beyond the observed window: one absorbing state.
    frame["_arrive"] = np.where(
        arrive.notna() & (arrive <= MAX_ARRIVAL), arrive, -1
    ).astype(int)
    frame.loc[frame["tier"] == "never", "_arrive"] = -1

    full_keys = ["player_type", "age", "sport_id", "age_rel_bucket", "perf_bucket"]
    back_keys = ["player_type", "age", "sport_id"]

    tables = []
    for keys, label in ((full_keys, "full"), (back_keys, "age_level")):
        counts = (
            frame.groupby(keys + ["tier", "_arrive"], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        totals = counts.groupby(keys, dropna=False)["n"].transform("sum")
        counts["p"] = counts["n"] / totals
        counts["cell_n"] = totals
        counts["conditioning"] = label
        if label == "age_level":
            counts["age_rel_bucket"] = pd.NA
            counts["perf_bucket"] = pd.NA
        tables.append(counts.rename(columns={"_arrive": "arrive"}))

    joint = pd.concat(tables, ignore_index=True)
    print(
        f"joint outcome table: {len(joint)} rows "
        f"({int((joint['conditioning'] == 'full').sum())} fully conditioned, "
        f"{int((joint['conditioning'] == 'age_level').sum())} backoff)"
    )
    return joint


def prospect_branches(
    joint: pd.DataFrame,
    archetypes: pd.DataFrame,
    player_type: str,
    age: int,
    sport_id: int,
    age_rel_bucket: str,
    perf_bucket: str,
    role: str,
    min_cell_n: int = MIN_CELL_N,
) -> tuple[list[dict], str]:
    """Outcome branches for one prospect, plus which conditioning was used.

    Returns:
        (branches, conditioning) where branches is the list
        `value.branch_payoffs` consumes. Lines are FULL-SEASON already, so pass
        them as the `later_branches` argument and give a prospect no season-0
        contribution — he is not in the majors now.

    Asserts when neither the conditioned cell nor the backoff has support. A
    prospect at an (age, level) the cohort never saw cannot be priced, and
    returning a default would be a fabricated prior on the exact players this
    model exists to judge.
    """
    lines = archetypes[archetypes["player_type"] == player_type].set_index("tier")
    assert not lines.empty, (
        f"prospect_branches: no archetype lines for {player_type!r}."
    )

    candidates = joint[
        (joint["player_type"] == player_type)
        & (joint["age"] == age)
        & (joint["sport_id"] == sport_id)
    ]
    full = candidates[
        (candidates["conditioning"] == "full")
        & (candidates["age_rel_bucket"] == age_rel_bucket)
        & (candidates["perf_bucket"] == perf_bucket)
    ]
    backoff = candidates[candidates["conditioning"] == "age_level"]

    if not full.empty and int(full["cell_n"].iloc[0]) >= min_cell_n:
        cell, conditioning = full, "full"
    else:
        assert not backoff.empty, (
            f"prospect_branches: no cohort support for {player_type} age {age} "
            f"at sport_id {sport_id}, at either conditioning. The cohort "
            f"(2005-2018) never observed that combination, so any prior would "
            f"be invented. Widen the level set or treat this player as "
            f"unscoreable and say so."
        )
        cell, conditioning = backoff, "age_level"

    total = float(cell["p"].sum())
    assert abs(total - 1.0) < 1e-6, (
        f"prospect_branches: cell probabilities sum to {total:.6f}, not 1.0, "
        f"for {player_type} age {age} sport {sport_id} ({conditioning}). The "
        f"never-arrives mass must be included or every prospect is silently "
        f"discounted."
    )

    branches = []
    for _, row in cell.iterrows():
        probability = float(row["p"])
        if probability <= 0.0:
            continue
        arrive = int(row["arrive"])
        tier = row["tier"]
        if arrive < 0 or tier == "never":
            # Never arrives: an all-zero line still has to carry its
            # probability mass, or branch_payoffs' sum-to-one assert fires.
            branches.append(
                {
                    "prob": probability,
                    "line": {stat: 0.0 for stat in LINE_STATS},
                    "arrive": 0,
                    "arrive_age": max(age, 20),
                    "role": role,
                }
            )
            continue
        arrive_age = age + arrive
        branches.append(
            {
                "prob": probability,
                "line": {
                    stat: float(lines.loc[tier, stat]) for stat in LINE_STATS
                },
                "arrive": arrive,
                # Aged from his age AT ARRIVAL, never from his current age: the
                # decay curves do not exist below 20, and a teenager aged from
                # today would be extrapolating a curve off data that cannot
                # exist.
                "arrive_age": int(np.clip(arrive_age, 20, 40)),
                "role": role,
            }
        )
    return branches, conditioning


def major_leaguer_branches(
    ros_line: dict[str, float],
    age: int,
    role: str,
    season_fraction_remaining: float,
) -> tuple[list[dict], list[dict]]:
    """The degenerate mixture: one branch, probability 1.

    Season 0 uses the rest-of-season line; later seasons use the annualized one.
    Kept here beside `prospect_branches` so the two populations are visibly the
    same abstraction rather than two code paths.
    """
    now = [
        {"prob": 1.0, "line": dict(ros_line), "arrive": 0,
         "arrive_age": int(np.clip(age, 20, 40)), "role": role}
    ]
    later = [
        {"prob": 1.0, "line": annualize(ros_line, season_fraction_remaining),
         "arrive": 0, "arrive_age": int(np.clip(age, 20, 40)), "role": role}
    ]
    return now, later
