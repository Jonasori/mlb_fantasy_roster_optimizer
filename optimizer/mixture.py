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
from scipy.optimize import isotonic_regression

from .value import annualize

# Below this many cohort observations a fully-conditioned cell is not usable and
# we back off to (age, level). Matches data_prep.prospect_outcomes.MIN_CELL_N.
MIN_CELL_N: int = 20

# Tiers, worst to best. "never" carries the whole non-arrival mass and must be
# present in every mixture or the probabilities silently under-sum.
TIERS: tuple[str, ...] = ("never", "fringe", "regular", "star")
# Tier ordering used by the monotonicity constraint and the score tilt. Index is
# the "how good is this outcome" rank, and both mechanisms assume it.
TIER_RANK: dict[str, int] = {tier: rank for rank, tier in enumerate(TIERS)}

# Strength of the within-cell score tilt (`tilt_tier_mass`). At 0.5, a prospect
# at the top of his pool (score +1) sees his star-versus-never odds multiplied
# by exp(0.5 * 1 * 3) ~= 4.5 against the cell base rate, and one at the bottom
# by the reciprocal. UNCALIBRATED -- it is a prior on how much a peak projection
# should be allowed to override a cohort base rate, and nothing has measured it.
# Report `dynasty.tilt_sensitivity` before leaning on any ranking it produces.
DEFAULT_TILT: float = 0.5

# Rookie-eligibility thresholds, which is the league's OWN definition of when a
# prospect stops being one. Used by `graduating_branches` to weight MLB evidence
# against cohort base rates, so the blend has a real-world anchor rather than an
# invented cut.
ROOKIE_PA: float = 130.0
ROOKIE_IP: float = 50.0

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
    # The INTERMEDIATE backoff keeps performance and drops age-relative-to-level,
    # which is the opposite of what the prospect literature's emphasis suggests
    # and is nevertheless right here: age_rel is nearly collinear with
    # (age, level) by construction — a 19-year-old in A-ball has age_rel of about
    # -1.8 whether or not anyone measures it — so conditioning on (age, level)
    # has already absorbed almost all of it. Performance is the signal that is
    # actually still free.
    #
    # Without this rung, dropping straight to (age, level) made five different
    # 22-year-old Double-A hitters score IDENTICALLY despite spanning the <90 to
    # 125+ performance range, which is useless for picking among prospects.
    perf_keys = ["player_type", "age", "sport_id", "perf_bucket"]
    back_keys = ["player_type", "age", "sport_id"]

    tables = []
    for keys, label in (
        (full_keys, "full"),
        (perf_keys, "age_level_perf"),
        (back_keys, "age_level"),
    ):
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
        if "age_rel_bucket" not in keys:
            counts["age_rel_bucket"] = pd.NA
        if "perf_bucket" not in keys:
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
    # Most specific first; take the first rung with enough support. Each rung
    # drops one conditioning variable, and the order is deliberate — see
    # joint_outcome_table on why performance outranks age-relative-to-level.
    rungs = (
        (
            "full",
            candidates[
                (candidates["conditioning"] == "full")
                & (candidates["age_rel_bucket"] == age_rel_bucket)
                & (candidates["perf_bucket"] == perf_bucket)
            ],
        ),
        (
            "age_level_perf",
            candidates[
                (candidates["conditioning"] == "age_level_perf")
                & (candidates["perf_bucket"] == perf_bucket)
            ],
        ),
        ("age_level", candidates[candidates["conditioning"] == "age_level"]),
    )
    cell, conditioning = None, None
    for label, candidate in rungs:
        if not candidate.empty and int(candidate["cell_n"].iloc[0]) >= min_cell_n:
            cell, conditioning = candidate, label
            break
    if cell is None:
        # Last rung regardless of size, but only if it exists at all.
        _, coarsest = rungs[-1]
        assert not coarsest.empty, (
            f"prospect_branches: no cohort support for {player_type} age {age} "
            f"at sport_id {sport_id}, at any conditioning. The cohort "
            f"(2005-2018) never observed that combination, so any prior would "
            f"be invented. Widen the level set or treat this player as "
            f"unscoreable and say so."
        )
        cell, conditioning = coarsest, "age_level"

    total = float(cell["p"].sum())
    assert abs(total - 1.0) < 1e-6, (
        f"prospect_branches: cell probabilities sum to {total:.6f}, not 1.0, "
        f"for {player_type} age {age} sport {sport_id} ({conditioning}). The "
        f"never-arrives mass must be included or every prospect is silently "
        f"discounted."
    )

    # SHRINK TOWARD THE LEVEL MARGINAL. A thin cell produces extreme
    # probabilities, and it does so in the most damaging possible direction:
    # measured on the live roster, 19-year-olds in Double-A came out with
    # P(never reaches MLB) = 0.000 off a handful of observations, i.e. the model
    # claimed a certainty of arrival for the rarest and most exciting prospects
    # on the board. Those are precisely the players a star-hunting strategy
    # would then overpay for.
    #
    # Standard empirical Bayes: weight the cell by n/(n+M) against the
    # (player_type, level) marginal. For a well-populated cell the adjustment is
    # negligible; for a thin one the marginal dominates. No special-casing, no
    # threshold to tune beyond M.
    cell_n = float(cell["cell_n"].iloc[0])
    marginal = joint[
        (joint["player_type"] == player_type)
        & (joint["sport_id"] == sport_id)
        & (joint["conditioning"] == "age_level")
    ]
    weight = cell_n / (cell_n + float(min_cell_n))
    prior = (
        marginal.groupby(["tier", "arrive"], dropna=False)["n"].sum()
        if not marginal.empty
        else None
    )
    observed = cell.set_index(["tier", "arrive"])["p"]
    if prior is not None and float(prior.sum()) > 0:
        prior_p = prior / float(prior.sum())
        blended = observed.mul(weight).add(prior_p.mul(1.0 - weight), fill_value=0.0)
        blended = blended / float(blended.sum())
        cell = (
            blended.rename("p")
            .reset_index()
            .assign(cell_n=cell_n, conditioning=conditioning)
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
                    "tier": "never",
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
                "tier": tier,
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
         "arrive_age": int(np.clip(age, 20, 40)), "role": role, "tier": "mlb"}
    ]
    later = [
        {"prob": 1.0, "line": annualize(ros_line, season_fraction_remaining),
         "arrive": 0, "arrive_age": int(np.clip(age, 20, 40)), "role": role,
         "tier": "mlb"}
    ]
    return now, later


# ==========================================================================
# MONOTONICITY
# ==========================================================================


def enforce_monotone_perf(
    joint: pd.DataFrame, perf_order: tuple[str, ...]
) -> pd.DataFrame:
    """Force the tier distribution to improve with the performance bucket.

    WHY THIS IS NOT OPTIONAL. Measured on the built tables, 37 of 55
    (age, level) strata were NON-MONOTONE in performance: a better
    level-normalized line produced a LOWER V. Taitn Gray scored 0.309 in the
    110-125 bucket and 0.250 in 125+; Jeter Martinez scored 0.059 at 110-125
    and 0.034 at 125+. That is thin-cell sampling noise surviving the n/(n+M)
    shrinkage, and a prospect model that is not monotone in prospect
    performance is indefensible on its face regardless of what else it gets
    right.

    THE CONSTRAINT is first-order stochastic dominance in the performance
    bucket. For each tier threshold j, the survival mass

        S_j(b) = P(tier >= j | bucket b)

    must be non-decreasing in b. Each S_j is isotonically regressed across the
    buckets, weighted by cell_n so a well-populated cell moves a thin one and
    not the reverse. Cross-threshold ordering (S_star <= S_regular <= S_fringe)
    is then restored by a cumulative minimum from the top, because per-threshold
    isotonic regression does not guarantee it.

    This ADDS no information. It only removes orderings the data cannot
    support, which is why it is safe to apply before any scouting signal.

    Arrival splits are preserved: within a tier the (tier, arrive) rows are
    rescaled by the same factor, so only the tier marginal moves. A tier that
    gains mass where it previously had none inherits the group's pooled arrival
    distribution for that tier — inventing one would be worse, and leaving the
    mass unplaced would break the sum-to-one contract.

    Args:
        joint: Output of `joint_outcome_table`.
        perf_order: Performance bucket labels, worst to best. Pass
            `data_prep.prospect_outcomes.PERF_LABELS`.

    Returns:
        The same frame with `p` adjusted. Rows whose conditioning does not
        carry a performance bucket ("age_level") pass through untouched.
    """
    assert len(perf_order) >= 2, (
        f"enforce_monotone_perf: need at least two performance buckets to "
        f"order, got {perf_order}."
    )
    rank = {label: index for index, label in enumerate(perf_order)}
    thresholds = TIERS[1:]  # never is P(tier >= never) = 1 by definition

    out = joint.copy()
    out["_adjusted"] = False
    for conditioning in ("full", "age_level_perf"):
        block = out[out["conditioning"] == conditioning]
        if block.empty:
            continue
        group_keys = ["player_type", "age", "sport_id"]
        if conditioning == "full":
            group_keys = group_keys + ["age_rel_bucket"]

        for _, group in block.groupby(group_keys, dropna=False):
            buckets = [b for b in perf_order if b in set(group["perf_bucket"])]
            if len(buckets) < 2:
                continue
            marginal = (
                group.groupby(["perf_bucket", "tier"], dropna=False)["p"]
                .sum()
                .unstack(fill_value=0.0)
                .reindex(index=buckets, columns=list(TIERS), fill_value=0.0)
            )
            weights = np.array(
                [
                    float(group.loc[group["perf_bucket"] == b, "cell_n"].iloc[0])
                    for b in buckets
                ],
                dtype=float,
            )
            # Survival masses, isotonic in the bucket order.
            survivals = {}
            for threshold in thresholds:
                columns = TIERS[TIER_RANK[threshold]:]
                observed = marginal[list(columns)].sum(axis=1).to_numpy(dtype=float)
                survivals[threshold] = isotonic_regression(
                    observed, weights=weights, increasing=True
                ).x
            # Restore S_star <= S_regular <= S_fringe, from the top down.
            running = np.ones(len(buckets), dtype=float)
            for threshold in thresholds:
                running = np.minimum(running, survivals[threshold])
                survivals[threshold] = running.copy()

            # Rebuild the tier marginal by differencing the survival masses.
            target = pd.DataFrame(0.0, index=buckets, columns=list(TIERS))
            upper = np.ones(len(buckets), dtype=float)
            for threshold in thresholds:
                target[TIERS[TIER_RANK[threshold] - 1]] = upper - survivals[threshold]
                upper = survivals[threshold]
            target[TIERS[-1]] = upper

            if np.allclose(
                target.to_numpy(dtype=float),
                marginal.to_numpy(dtype=float),
                atol=1e-12,
            ):
                continue

            # Pooled arrival split per tier, for a tier that gains mass from
            # nothing and therefore has no arrival distribution of its own.
            pooled = group.groupby(["tier", "arrive"], dropna=False)["p"].sum()
            for bucket in buckets:
                rows = group[group["perf_bucket"] == bucket]
                for tier in TIERS:
                    want = float(target.loc[bucket, tier])
                    have = float(marginal.loc[bucket, tier])
                    index = rows.index[rows["tier"] == tier]
                    if have > 0.0:
                        out.loc[index, "p"] = out.loc[index, "p"] * (want / have)
                        out.loc[index, "_adjusted"] = True
                        continue
                    if want <= 0.0 or len(index) > 0:
                        continue
                    # No rows for this tier in this bucket: place the new mass
                    # on the group's pooled arrival split for that tier.
                    if tier in pooled.index.get_level_values("tier"):
                        split = pooled.loc[tier]
                        split = split / float(split.sum())
                    else:
                        split = pd.Series({-1: 1.0})
                    template = rows.iloc[0]
                    for arrive, share in split.items():
                        new = template.copy()
                        new["tier"] = tier
                        new["arrive"] = int(arrive) if tier != "never" else -1
                        new["p"] = want * float(share)
                        new["_adjusted"] = True
                        out.loc[len(out)] = new

    n_adjusted = int(out["_adjusted"].sum())
    print(
        f"enforce_monotone_perf: adjusted {n_adjusted} of {len(out)} rows to "
        f"restore stochastic ordering in the performance bucket"
    )
    return out.drop(columns="_adjusted").reset_index(drop=True)


def tilt_tier_mass(
    branches: list[dict], score: float, strength: float = DEFAULT_TILT
) -> list[dict]:
    """Re-weight a prospect's outcome tiers by a within-cell quality score.

    THE PROBLEM THIS SOLVES. Every prospect in a conditioning cell currently
    receives an IDENTICAL score, because the cell is all the model knows about
    him. Measured on the live pool: 44 cells hold 4 or more players, covering
    277 of 433 prospects, and the within-cell spread of our V is exactly 0.000
    by construction while OOPSY's peak projection spreads those same players
    over a median of 5.57 z-units. Eleven 23-year-old AAA hitters in the
    110-125 bucket all scored 0.4328 — Emmanuel Rodriguez and Ryan Waldschmidt
    alike.

    THE MECHANISM is exponential tilting on the tier rank:

        p'_k  ∝  p_k · exp(strength · score · rank(tier_k))

    which is the standard monotone-likelihood-ratio family. Two properties make
    it the right choice over anything ad hoc:

      * It is MONOTONE in `score` by construction. A better score can never
        lower the mass on a better tier relative to a worse one, so it cannot
        reintroduce the inversion `enforce_monotone_perf` just removed.
      * It preserves the ARRIVAL distribution within each tier, and it cannot
        create mass on a tier the cell never observed. The cohort still decides
        what outcomes are possible and when they land; the score only decides
        how the mass is distributed among them.

    `score` must already be centered on the comparison pool, so that score = 0
    returns the cell's own base rate unchanged. Feeding a raw projection here
    would tilt every prospect in the same direction and change nothing about
    their ordering while corrupting the absolute level.

    Args:
        branches: From `prospect_branches`. Each must carry a "tier" key.
        score: Centered quality score, positive meaning better than cellmates.
        strength: Tilt coefficient. 0.0 disables the tilt exactly.

    Returns:
        A new branch list with the same lines, arrivals and roles, and
        probabilities that still sum to 1.
    """
    assert strength >= 0.0, (
        f"tilt_tier_mass: strength must be >= 0, got {strength}. A negative "
        f"value would make a better score lower the star mass."
    )
    if strength == 0.0 or score == 0.0:
        return [dict(branch) for branch in branches]
    for branch in branches:
        assert "tier" in branch, (
            "tilt_tier_mass: a branch has no 'tier' key, so the tilt has "
            "nothing to act on. Build branches with prospect_branches."
        )
        assert branch["tier"] in TIER_RANK, (
            f"tilt_tier_mass: unknown tier {branch['tier']!r}. Only a PROSPECT "
            f"mixture can be tilted; a major leaguer's certain branch has "
            f"nothing to re-weight."
        )

    weights = np.array(
        [
            float(branch["prob"])
            * float(np.exp(strength * score * TIER_RANK[branch["tier"]]))
            for branch in branches
        ],
        dtype=float,
    )
    total = float(weights.sum())
    assert total > 0.0, (
        f"tilt_tier_mass: tilted weights sum to {total}. The score "
        f"({score}) or strength ({strength}) has underflowed every branch; "
        f"clip the score to a sane range before calling."
    )
    weights = weights / total
    return [
        {**branch, "prob": float(weight)}
        for branch, weight in zip(branches, weights)
    ]


def graduation_weight(
    mlb_pa: float, mlb_ip: float, player_type: str
) -> float:
    """How much of a player's future is settled by his MLB record so far.

    0.0 = no major-league evidence, price him entirely off the cohort.
    1.0 = past rookie eligibility, price him entirely off his projection.

    The threshold is ROOKIE_PA / ROOKIE_IP, which is the league's own
    definition of when a prospect stops being one. Using it means the blend is
    anchored to a real rule rather than a number chosen to make the output look
    reasonable.
    """
    assert player_type in ("hitter", "pitcher"), (
        f"graduation_weight: player_type must be 'hitter' or 'pitcher', got "
        f"{player_type!r}."
    )
    assert mlb_pa >= 0.0 and mlb_ip >= 0.0, (
        f"graduation_weight: volumes must be non-negative, got PA={mlb_pa}, "
        f"IP={mlb_ip}."
    )
    if player_type == "hitter":
        return float(min(1.0, mlb_pa / ROOKIE_PA))
    return float(min(1.0, mlb_ip / ROOKIE_IP))


def graduating_branches(
    prospect_later: list[dict],
    mlb_now: list[dict],
    mlb_later: list[dict],
    weight: float,
) -> tuple[list[dict], list[dict]]:
    """Blend a prospect's cohort mixture with his own major-league projection.

    THE BUG THIS FIXES. Routing used to be a hard test — a player with fewer
    than 20 projected PA and 5 projected IP took the prospect path, everyone
    else took the major-league path. That threshold silently chose between two
    estimators that disagree by up to 5x. Kade Anderson, a 21-year-old in
    Double-A at the top performance bucket, drew an 11-inning rest-of-season
    projection, which was enough to send him down the major-league path and
    value him at 0.043 against the 0.217 his cohort supports. A projection
    system handing a prospect token volume is not evidence that he has
    graduated, and treating it as such is how a post-hype player gets dropped
    on the way up. 343 of the 433 players carrying a live minor-league line
    were being routed this way.

    THE FIX is to stop choosing. Graduation is continuous, so the player is a
    mixture of "already arrived, and here is the projection" at `weight` and
    "still a prospect, and here is the cohort" at 1 - weight. There is no
    threshold left to be wrong about, and both extremes recover the old
    behaviour exactly.

    Season 0 is handled separately because a not-yet-arrived branch contributes
    nothing now: the prospect mass appears in season 0 as an explicit zero line
    carrying probability 1 - weight, which keeps the sum-to-one contract that
    `value.branch_payoffs` asserts on.

    Args:
        prospect_later: From `prospect_branches`. Full-season lines.
        mlb_now, mlb_later: From `major_leaguer_branches`.
        weight: From `graduation_weight`, in [0, 1].

    Returns:
        (now_branches, later_branches), each summing to probability 1.
    """
    assert 0.0 <= weight <= 1.0, (
        f"graduating_branches: weight must lie in [0, 1], got {weight}. Use "
        f"graduation_weight, which clamps it."
    )
    assert len(mlb_now) == len(mlb_later) == 1, (
        f"graduating_branches: expected one certain major-league branch on "
        f"each horizon, got {len(mlb_now)} and {len(mlb_later)}."
    )
    zero_line = {stat: 0.0 for stat in LINE_STATS}

    # now and later are returned BRANCH-ALIGNED: same length, same order, same
    # probabilities. Only the lines differ, because season 0 is the remainder of
    # this season and every later season is a whole one. Callers splice column 0
    # of a per-branch matrix from `now`, which is only well defined if the two
    # lists correspond element for element.
    now = [{**mlb_now[0], "prob": weight}]
    later = [{**mlb_later[0], "prob": weight}]
    for branch in prospect_later:
        probability = float(branch["prob"]) * (1.0 - weight)
        later.append({**branch, "prob": probability})
        # A prospect contributes nothing in season 0 by construction: he is not
        # in the majors now. The zero line carries his mass so the sum-to-one
        # contract holds on both horizons.
        now.append(
            {
                **branch,
                "prob": probability,
                "line": dict(zero_line),
                "arrive": 0,
            }
        )
    return now, later
