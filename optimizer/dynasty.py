"""
The dynasty valuation, end to end: data on disk in, V(p; β) and break-evens out.

This is the entry point the rest of the stack was missing. `championship`,
`value` and `mixture` are libraries with no opinion about where their inputs
come from; this module holds the opinions — which snapshot, which population,
which alternative — in one readable place.

    context = season_context(state)
    frame, payoffs = score_universe(players, context, ...)
    compare(payoffs, "Trea Turner", "Devin Taylor")

THREE THINGS IT FIXES RELATIVE TO THE SCRATCH DRIVER IT REPLACES
----------------------------------------------------------------
1. GRADUATION IS CONTINUOUS. The old driver picked the prospect path or the
   major-league path on a 20-PA threshold, which silently chose between two
   estimators that disagreed by 5x for Kade Anderson and by 10x the other way
   for Jake Bennett. 343 of 433 players carrying a live minor-league line were
   routed on it. Now `mixture.graduation_weight` blends them against rookie
   eligibility, and there is no threshold left to be wrong about.

2. WITHIN-CELL DISCRIMINATION. Every prospect in a conditioning cell used to
   score identically — 277 of 433 sat in a cell with at least four members, and
   the within-cell spread of V was exactly zero while OOPSY's peak projection
   spread the same players over 5.6 z-units. `mixture.tilt_tier_mass` re-weights
   the tier mass by a centered within-pool score.

3. THE HORIZON IS NOT 8. The scratch driver scored 8 seasons, which truncated
   exactly the players the model exists to find: Ethan Holliday's payoff was
   still 0.103 in the last scored season and flat, so his tail was simply
   discarded and Manny Machado "dominated" him by 4%. A gradient costs 1.8
   seconds, so 8 was never a cost decision. DEFAULT_N_SEASONS covers the
   8-year arrival window plus a productive career on top of it.

POSITION ENTERS HERE AND NOWHERE ELSE. A category line does not know what glove
its owner wears, so scoring stays position-blind by construction. Scarcity is a
statement about the ALTERNATIVE, not about the player, and it belongs in the
opportunity-cost difference — see `slot_alternatives`.
"""

import numpy as np
import pandas as pd

from .championship import build_season_context
from .config import (
    MINOR_LEAGUE_SLOTS,
    MY_TEAM_NAME,
    SLOT_ELIGIBILITY,
    season_fraction_remaining,
)
from .mixture import (
    DEFAULT_TILT,
    graduating_branches,
    graduation_weight,
    major_leaguer_branches,
    prospect_branches,
    tilt_tier_mass,
)
from .value import (
    LINE_STATS,
    best_available,
    branch_payoff_matrix,
    branch_payoffs,
    breakeven_beta,
    dominates,
    player_role,
    player_value,
    stopped_value,
)

# The arrival window is 8 seasons (mixture.MAX_ARRIVAL) and a player who arrives
# at the end of it still has a career. Anything shorter truncates the late-
# arriving prospects the model is for, and `survival` drives the tail to zero on
# its own well before this.
DEFAULT_N_SEASONS: int = 16

# Betas reported by default. Spanning the full range is deliberate: no number
# from this model means anything without a beta attached, so the report shows
# the sweep rather than inviting a single quotation.
DEFAULT_BETAS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)


def season_context(state: dict, n_seasons: int = DEFAULT_N_SEASONS, **kwargs) -> dict:
    """Championship gradients per season, from a `compute_league_state` result."""
    for key in ("my_totals", "opponent_totals", "category_sigmas"):
        assert key in state, (
            f"season_context: state is missing {key!r}. Pass the dict returned "
            f"by optimizer.league_state.compute_league_state."
        )
    return build_season_context(
        state["my_totals"], state["opponent_totals"], state["category_sigmas"],
        n_seasons, **kwargs,
    )


def pool_scores(
    peak: pd.DataFrame, ids: set[int], player_type: str
) -> dict[int, float]:
    """Centered within-pool quality score in [-1, 1], for `tilt_tier_mass`.

    Rank-based, not a z-score. The peak projection's scale is arbitrary and its
    tail is heavy, so a raw z would let one outlier dominate the tilt; a
    percentile is bounded by construction and the median maps to exactly 0,
    which is what makes an untilted player recover his cell's base rate.

    Scored against the PROSPECT POOL rather than all of baseball, because the
    tilt's job is to order players inside a cohort cell — comparing a Single-A
    arm to Tarik Skubal would push every prospect to the same extreme and order
    none of them.

    Args:
        peak: OOPSY peak snapshot, carrying MLBAMID and an 'FV' column.
        ids: MLBAM ids of the pool to rank within.
        player_type: "hitter" or "pitcher"; ranked separately because FV is
            standardised within type.

    Returns:
        {mlbam_id: score}. Players absent from the peak feed are absent here,
        and `score_universe` leaves them untilted rather than guessing.
    """
    assert "FV" in peak.columns, (
        "pool_scores: peak frame needs an 'FV' column. Run it through "
        "optimizer.player_scoring.add_fantasy_value first."
    )
    block = peak[
        (peak["player_type"] == player_type) & peak["MLBAMID"].isin(ids)
    ].dropna(subset=["MLBAMID", "FV"])
    if len(block) < 2:
        print(
            f"pool_scores: only {len(block)} {player_type}s in the pool carry a "
            f"peak projection; skipping the tilt for that side."
        )
        return {}
    percentile = block["FV"].rank(pct=True)
    return {
        int(mlbam): float(2.0 * value - 1.0)
        for mlbam, value in zip(block["MLBAMID"], percentile)
    }


def slot_alternatives(
    payoffs: dict[str, np.ndarray],
    frame: pd.DataFrame,
    slot: str,
    beta: float,
    owner_column: str = "owner",
) -> tuple[str | None, np.ndarray]:
    """Best UNOWNED player eligible at `slot`, and his payoff stream.

    This is the only place position affects value. A single-catcher league has
    a far thinner catcher pool than outfield pool, so the same production is
    worth more behind the plate — and that premium falls out of the difference
    rather than being asserted as a positional adjustment.

    A slot with no eligible free agent returns (None, zeros), which correctly
    prices it as irreplaceable.
    """
    assert slot in SLOT_ELIGIBILITY, (
        f"slot_alternatives: {slot!r} is not a configured slot. Known: "
        f"{sorted(SLOT_ELIGIBILITY)}."
    )
    eligible = SLOT_ELIGIBILITY[slot]
    free = frame[frame[owner_column].isna() & frame["Name"].isin(payoffs)]
    names = [
        row["Name"]
        for _, row in free.iterrows()
        if isinstance(row["pos"], str)
        and eligible & {p.strip() for p in row["pos"].split(",")}
    ]
    return best_available(payoffs, names, beta)


def score_universe(
    players: pd.DataFrame,
    context: dict,
    milb_state: pd.DataFrame,
    joint: pd.DataFrame,
    archetypes: pd.DataFrame,
    decay_lookup,
    survival_lookup,
    mlb_volume: pd.DataFrame | None = None,
    peak: pd.DataFrame | None = None,
    tilt: float = DEFAULT_TILT,
    betas: tuple[float, ...] = DEFAULT_BETAS,
    fraction_remaining: float | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, tuple]]:
    """Score every player, blending cohort and projection by graduation weight.

    Args:
        players: data/players.parquet.
        context: From `season_context`.
        milb_state: data/priors/milb_2026_state.parquet.
        joint, archetypes: The cohort tables, monotonicity already enforced.
        mlb_volume: Season-to-date MLB PA/IP by MLBAMID, for the graduation
            weight. None means every prospect is treated as ungraduated, which
            is the conservative direction but understates anyone who has
            already established himself.
        peak: OOPSY peak snapshot with FV, for the within-cell tilt. None
            disables the tilt.
        tilt: Passed to `mixture.tilt_tier_mass`. 0.0 disables it exactly.

    Returns:
        (frame, payoffs, matrices). `payoffs[name]` is the expected per-season
        payoff vector; `matrices[name]` is (probs, branch matrix) for
        `value.stopped_value`.
    """
    if fraction_remaining is None:
        fraction_remaining = season_fraction_remaining()
    n_seasons = len(context["gradients"])
    in_pool = (
        milb_state[milb_state["in_pool"]]
        .sort_values("sport_id")
        .drop_duplicates("player_id", keep="first")
        .set_index("player_id")
    )
    volume = (
        mlb_volume.set_index("MLBAMID") if mlb_volume is not None else None
    )
    scores: dict[str, dict[int, float]] = {}
    if peak is not None and tilt > 0.0:
        ids = set(in_pool.index.astype(int))
        for player_type in ("hitter", "pitcher"):
            scores[player_type] = pool_scores(peak, ids, player_type)

    rows, payoffs, matrices, skipped = [], {}, {}, []
    for _, player in players.iterrows():
        name = player["Name"]
        player_type = player["player_type"]
        mlbam = player["MLBAMID"]
        milb = (
            in_pool.loc[int(mlbam)]
            if pd.notna(mlbam) and int(mlbam) in in_pool.index
            else None
        )

        age = player["age"]
        position = player["Position"]
        if not np.isfinite(age):
            skipped.append((name, "no age")); continue
        if player_type == "pitcher" and not isinstance(position, str):
            skipped.append((name, "pitcher with no Position")); continue
        role = player_role(position, player_type)

        # How much of his future is already settled by his MLB record?
        if milb is None:
            weight = 1.0
        elif volume is not None and pd.notna(mlbam) and int(mlbam) in volume.index:
            banked = volume.loc[int(mlbam)]
            weight = graduation_weight(
                float(banked.get("PA", 0.0)), float(banked.get("IP", 0.0)),
                player_type,
            )
        else:
            weight = 0.0

        line = {stat: float(player[stat]) for stat in LINE_STATS}
        mlb_now = mlb_later = None
        if weight > 0.0:
            if int(age) < 20:
                skipped.append((name, "MLB evidence but age below fitted band"))
                continue
            mlb_now, mlb_later = major_leaguer_branches(
                line, int(min(age, 40)), role, fraction_remaining
            )

        prospect_later, conditioning, score = None, "certain", 0.0
        if weight < 1.0:
            assert milb is not None, (
                f"score_universe: {name} has graduation weight {weight} but no "
                f"minor-league row. That combination is impossible; the weight "
                f"is set to 1.0 whenever milb is None."
            )
            prospect_role = "hitter" if player_type == "hitter" else "starter"
            prospect_later, conditioning = prospect_branches(
                joint, archetypes, player_type, int(milb["age"]),
                int(milb["sport_id"]), milb["age_rel_bucket"],
                milb["perf_bucket"], prospect_role,
            )
            score = scores.get(player_type, {}).get(int(mlbam), 0.0)
            if tilt > 0.0 and score != 0.0:
                prospect_later = tilt_tier_mass(prospect_later, score, tilt)

        if mlb_now is None:
            # No major-league mass at all: a placeholder certain branch supplies
            # the age and role, and graduating_branches gives it weight 0.
            age_now = int(np.clip(milb["age"], 20, 40))
            mlb_now, mlb_later = major_leaguer_branches(
                line, age_now, role, fraction_remaining
            )
        if prospect_later is None:
            prospect_later = []
        now, later = graduating_branches(
            prospect_later, mlb_now, mlb_later, weight
        )

        # Season 0 is the REMAINDER of this season, every later season a whole
        # one, so the two horizons need separate lines. `now` and `later` are
        # branch-aligned, which is what makes splicing column 0 well defined.
        _, head_matrix = branch_payoff_matrix(
            now, context["gradients"][:1], context["references"][:1],
            decay_lookup, survival_lookup, 1,
        )
        probs, matrix = branch_payoff_matrix(
            later, context["gradients"], context["references"],
            decay_lookup, survival_lookup, n_seasons,
        )
        matrix[:, 0] = head_matrix[:, 0]
        payoff = probs @ matrix

        payoffs[name] = payoff
        matrices[name] = (probs, matrix)
        record = {
            "Name": name, "age": int(age), "type": player_type, "role": role,
            "pos": position, "owner": player["owner"],
            "status": player["roster_status"], "grad_w": weight,
            "cond": conditioning, "score": score,
            "level": milb["level"] if milb is not None else "MLB",
            "u0": float(payoff[0]),
        }
        for beta in betas:
            record[f"V@{beta:g}"] = player_value(payoff, beta)
            record[f"S@{beta:g}"] = stopped_value(probs, matrix, beta)
        rows.append(record)

    frame = pd.DataFrame(rows)
    print(
        f"score_universe: scored {len(frame)} of {len(players)}; "
        f"{int((frame['grad_w'] < 1.0).sum())} carry cohort mass, "
        f"{int(((frame['grad_w'] > 0) & (frame['grad_w'] < 1)).sum())} are "
        f"mid-graduation; skipped {len(skipped)}"
    )
    if skipped:
        from collections import Counter
        print("  skips:", Counter(reason for _, reason in skipped).most_common())
    return frame, payoffs, matrices


def compare(
    payoffs: dict[str, np.ndarray], name_a: str, name_b: str
) -> dict:
    """Head-to-head: dominance first, then the break-even beta if there is one.

    Dominance is reported FIRST and separately because it is parameter-free —
    if it holds, no argument about impatience can reverse the ranking and there
    is nothing left to calibrate. Only when it fails does beta matter at all.
    """
    for name in (name_a, name_b):
        assert name in payoffs, (
            f"compare: {name!r} has no payoff vector. Available names come "
            f"from score_universe; check the -H/-P suffix."
        )
    a, b = payoffs[name_a], payoffs[name_b]
    result = breakeven_beta(a, b)
    result["a"], result["b"] = name_a, name_b
    if dominates(a, b):
        result["verdict"] = f"{name_a} DOMINATES {name_b} — true at every beta"
    elif dominates(b, a):
        result["verdict"] = f"{name_b} DOMINATES {name_a} — true at every beta"
    elif result["unique"]:
        result["verdict"] = (
            f"hold {name_a} over {name_b} only if beta < {result['roots'][0]:.3f}"
        )
    elif not result["roots"]:
        leader = name_a if player_value(a, 1.0) > player_value(b, 1.0) else name_b
        result["verdict"] = f"{leader} wins at every beta in (0, 1], without dominating"
    else:
        result["verdict"] = (
            f"{result['sign_changes']} sign changes, crossings at "
            f"{[round(r, 3) for r in result['roots']]} — a single break-even "
            f"would misrepresent this pair"
        )
    return result


def breakeven_board(
    payoffs: dict[str, np.ndarray], veterans: list[str], prospects: list[str]
) -> pd.DataFrame:
    """Every veteran-versus-prospect break-even, as one table.

    The report the model is actually for: not "who is better", which needs a
    beta nobody has chosen, but "how patient would you have to be for this
    trade to flip".
    """
    rows = []
    for veteran in veterans:
        for prospect in prospects:
            result = compare(payoffs, veteran, prospect)
            rows.append(
                {
                    "veteran": veteran, "prospect": prospect,
                    "breakeven": result["roots"][0] if result["unique"]
                    and result["roots"] else np.nan,
                    "unique": result["unique"],
                    "verdict": result["verdict"],
                }
            )
    return pd.DataFrame(rows).sort_values("breakeven", na_position="last")
