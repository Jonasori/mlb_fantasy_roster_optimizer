"""Backtest harness for projection corrections.

Assembles (projection-at-D, evidence-through-D, actual-after-D) triples and
scores candidate correctors. Nothing in this module ships to the optimizer;
it exists so that no correction reaches production on a tuned constant.

Per the spec §3.1, the evidence side is cheap: byDateRange is a league-wide
leaderboard, so any split date costs two requests.
"""

import datetime

import numpy as np
import pandas as pd

from data_prep.skills import HITTING_SKILLS, PITCHING_SKILLS, add_skill_rates
from data_prep.statsapi_stats import fetch_stats_range

# Regular-season end dates. Used to bound the outcome window.
SEASON_END: dict[int, datetime.date] = {
    2021: datetime.date(2021, 10, 3),
    2022: datetime.date(2022, 10, 5),
    2023: datetime.date(2023, 10, 1),
    2024: datetime.date(2024, 9, 29),
    2025: datetime.date(2025, 9, 28),
    2026: datetime.date(2026, 9, 27),
}

# Scoring columns, by side. These mirror the optimizer's category set.
HITTING_STATS: tuple[str, ...] = ("PA", "R", "HR", "RBI", "SB", "OPS")
PITCHING_STATS: tuple[str, ...] = ("IP", "W", "SV", "K", "ERA", "WHIP")


def _derive_outcome_stats(actual: pd.DataFrame, group: str) -> pd.DataFrame:
    """Compute the optimizer's scoring categories from raw outcome counts."""
    out = actual.copy()
    if group == "hitting":
        out["OPS"] = (out["obp"] + out["slg"]).astype(float)
        return out[["MLBAMID", *HITTING_STATS]]

    innings = out["IP"].where(out["IP"] > 0)
    out["ERA"] = (out["ER"] * 9.0 / innings).astype(float)
    out["WHIP"] = ((out["HA"] + out["BBA"]) / innings).astype(float)
    out["K"] = out["SOA"].astype(float)
    return out[["MLBAMID", *PITCHING_STATS]]


def assemble_backtest_frame(
    season: int,
    split: datetime.date,
    projection: pd.DataFrame,
    group: str = "hitting",
    evidence: pd.DataFrame | None = None,
    actual: pd.DataFrame | None = None,
    outcome_end: datetime.date | None = None,
) -> pd.DataFrame:
    """Build one (projection, evidence, outcome) triple for a split date.

    Args:
        season: Season year.
        split: The cut date. Evidence is season-start..split; the outcome
            window is split+1..outcome_end.
        projection: A dated rest-of-season projection as of `split`. Must
            carry MLBAMID plus the scoring columns for `group`.
        group: "hitting" or "pitching".
        evidence: Pre-fetched evidence frame. Fetched from StatsAPI if None.
        actual: Pre-fetched outcome frame. Fetched from StatsAPI if None.
        outcome_end: Last date of the outcome window. Defaults to the season
            end. A rest-of-season projection made at `split` projects all the
            way to the season end, so any shorter outcome window covers only
            part of what was projected — pass one explicitly (and see
            `proj_horizon_frac`) when scoring a partial, completed window.

    Returns:
        One row per player present in all three sources. Adds columns
        prefixed `proj_`, `evid_`, `actual_`, `n_evid` (the evidence sample
        size, PA for hitters and BF for pitchers), and `proj_horizon_frac`
        (the fraction of the projection's full split-to-season-end horizon
        that the outcome window actually covers; 1.0 unless `outcome_end` is
        set short of the season end).

        Hitting also carries evid_R/HR/RBI/SB (raw observed counts) and
        evid_OPS, so a baseline can re-project observed rates. Pitching
        carries the equivalent evid_W/SV/K/ERA/WHIP, plus n_evid_ip: n_evid
        for pitchers is BF (batters faced), which is the wrong denominator
        for scaling counting stats by innings, so evidence-window IP is
        carried separately.
    """
    assert group in ("hitting", "pitching"), (
        f"assemble_backtest_frame: group must be 'hitting' or 'pitching', got {group!r}."
    )
    dup_proj_ids = sorted(
        projection.loc[projection["MLBAMID"].duplicated(keep=False), "MLBAMID"].unique()
    )
    assert not dup_proj_ids, (
        f"assemble_backtest_frame: projection has duplicate MLBAMIDs "
        f"{dup_proj_ids} — a traded player split into two team rows (a real "
        f"FanGraphs export pattern) must be combined into one row before this "
        f"call, or the join below would silently keep an arbitrary one."
    )
    assert season in SEASON_END, (
        f"assemble_backtest_frame: no season end date recorded for {season}. "
        f"Known: {sorted(SEASON_END)}. Add it to SEASON_END."
    )
    end = SEASON_END[season]
    start = datetime.date(season, 1, 1)
    assert start < split < end, (
        f"assemble_backtest_frame: split date {split} must fall strictly "
        f"inside the {season} season ({start}..{end}). Outside it, one of the "
        f"two windows is empty and the backtest silently scores nothing."
    )
    if outcome_end is None:
        outcome_end = end
    today = datetime.date.today()
    assert outcome_end <= today, (
        f"assemble_backtest_frame: outcome_end {outcome_end} is in the future "
        f"(today is {today}) — that window has not finished, so the outcome "
        f"data would be right-censored. Pass an explicit past outcome_end, or "
        f"use a completed season."
    )
    assert outcome_end > split, (
        f"assemble_backtest_frame: outcome_end {outcome_end} is not after "
        f"split {split} — the outcome window would be empty or negative. "
        f"This bypasses fetch_stats_range's own start<=end check whenever "
        f"evidence/actual are injected rather than fetched, so it must be "
        f"checked here too."
    )

    if evidence is None:
        evidence = fetch_stats_range(season, start, split)
    if actual is None:
        actual = fetch_stats_range(
            season, split + datetime.timedelta(days=1), outcome_end
        )

    evidence = add_skill_rates(evidence[evidence["group"] == group])
    actual_raw = actual[actual["group"] == group].copy()

    skills = HITTING_SKILLS if group == "hitting" else PITCHING_SKILLS
    n_col = "n_PA" if group == "hitting" else "n_BF"
    stats = HITTING_STATS if group == "hitting" else PITCHING_STATS

    evid = evidence[["MLBAMID", *skills, n_col]].rename(
        columns={s: f"evid_{s}" for s in skills} | {n_col: "n_evid"}
    )
    if group == "pitching":
        # GB_pct and HRFB are computed over batted balls, not batters faced —
        # on real data n_BIP is a median 0.465 of n_BF and as low as 0.32 for
        # strikeout-heavy arms. Carrying only n_evid (= n_BF) would let later
        # shrinkage over-trust those two rates by more than 2x.
        evid["n_evid_bip"] = evidence["n_BIP"].astype(float).values
        # Raw observed pitching counts/rates, so the raw_ytd baseline can
        # project them onto the projection's volume — the pitching mirror of
        # evid_R/HR/RBI/SB below. Without these it silently degenerates into
        # the ATC baseline, same failure as the hitting side.
        evid["evid_W"] = evidence["W"].astype(float).values
        evid["evid_SV"] = evidence["SV"].astype(float).values
        evid["evid_K"] = evidence["SOA"].astype(float).values
        # ERA/WHIP derived exactly as _derive_outcome_stats does for actuals.
        evid_innings = evidence["IP"].where(evidence["IP"] > 0)
        evid["evid_ERA"] = (evidence["ER"] * 9.0 / evid_innings).astype(float).values
        evid["evid_WHIP"] = (
            (evidence["HA"] + evidence["BBA"]) / evid_innings
        ).astype(float).values
        # n_evid (= BF) is the wrong denominator for scaling W/SV/K by
        # innings, so evidence-window IP is carried under its own name.
        evid["n_evid_ip"] = evidence["IP"].astype(float).values
    if group == "hitting":
        # The volume corrector's slump term needs the observed composite, even
        # though every *rate* consumer downstream must use the components.
        evid["evid_OPS"] = (
            evidence["obp"].astype(float) + evidence["slg"].astype(float)
        ).values
        # Raw observed counting stats, so the raw_ytd baseline can project them
        # onto the projection's volume. Without these it silently degenerates
        # into the ATC baseline with one column swapped.
        for stat in ("R", "HR", "RBI", "SB"):
            evid[f"evid_{stat}"] = evidence[stat].astype(float).values
    proj = projection[["MLBAMID", *stats]].rename(
        columns={s: f"proj_{s}" for s in stats}
    )
    out = _derive_outcome_stats(actual_raw, group).rename(
        columns={s: f"actual_{s}" for s in stats}
    )

    name_col = "Name" if "Name" in projection.columns else "name"
    names = projection[["MLBAMID", name_col]].rename(columns={name_col: "name"})

    frame = (
        proj.merge(evid, on="MLBAMID", how="inner")
        .merge(out, on="MLBAMID", how="inner")
        .merge(names, on="MLBAMID", how="left")
    )
    dup_ids = sorted(
        frame.loc[frame["MLBAMID"].duplicated(keep=False), "MLBAMID"].unique()
    )
    assert not dup_ids, (
        f"assemble_backtest_frame: duplicate MLBAMIDs survived the join: "
        f"{dup_ids}. Each player must appear once in projection, evidence, "
        f"and actual — picking one at random would silently drop a "
        f"traded/split row. Check the source(s) for these IDs."
    )

    # Projected volume must be multiplied by this before it is compared to
    # actual volume — a rest-of-season projection covers split..season-end,
    # and a shorter outcome window only realizes part of that.
    frame["proj_horizon_frac"] = (outcome_end - split).days / (end - split).days

    assert len(frame) > 0, (
        f"assemble_backtest_frame: no players survived the join for {season} "
        f"@ {split} ({group}). Check that the projection carries MLBAMID and "
        f"that it is not the opposite player type."
    )
    print(
        f"backtest frame {season} @ {split} ({group}): {len(frame)} players "
        f"(projection {len(proj)}, evidence {len(evid)}, outcome {len(out)})"
    )
    return frame


def _mew_contribution(
    frame: pd.DataFrame, prefix: str, my_totals: dict, gradient: dict, group: str
) -> pd.Series:
    """MEW for each player from one column family (proj_, pred_, or actual_).

    Mirrors optimizer.player_scoring.add_mew exactly: counting stats enter
    linearly, ratio stats enter volume-weighted against the team's own rate.
    """
    if group == "hitting":
        volume = frame[f"{prefix}PA"].astype(float)
        mew = (
            gradient["R"] * frame[f"{prefix}R"]
            + gradient["HR"] * frame[f"{prefix}HR"]
            + gradient["RBI"] * frame[f"{prefix}RBI"]
            + gradient["SB"] * frame[f"{prefix}SB"]
        )
        mew = mew + gradient["OPS"] * volume * (
            frame[f"{prefix}OPS"] - my_totals["OPS"]
        ) / my_totals["PA"]
        return mew.astype(float)

    volume = frame[f"{prefix}IP"].astype(float)
    mew = (
        gradient["W"] * frame[f"{prefix}W"]
        + gradient["SV"] * frame[f"{prefix}SV"]
        + gradient["K"] * frame[f"{prefix}K"]
    )
    for cat in ("ERA", "WHIP"):
        mew = mew + gradient[cat] * volume * (
            frame[f"{prefix}{cat}"] - my_totals[cat]
        ) / my_totals["IP"]
    return mew.astype(float)


def score_in_mew(
    frame: pd.DataFrame, my_totals: dict, gradient: dict, group: str = "hitting"
) -> pd.DataFrame:
    """Score a prediction in MEW units — the only metric that decides anything.

    Requires columns: pred_<stat> and actual_<stat> for every scoring category
    of `group`.
    Adds columns: mew_pred, mew_actual, mew_error.

    Stat-unit error is reported alongside by the caller, but the spec's §3.1
    decision rule is this column: a method that cuts OPS RMSE by 20% while
    moving no decision has earned nothing.
    """
    frame = frame.copy()
    stats = HITTING_STATS if group == "hitting" else PITCHING_STATS
    for stat in stats:
        for prefix in ("pred_", "actual_"):
            col = f"{prefix}{stat}"
            assert col in frame.columns, (
                f"score_in_mew: missing column {col}. Every candidate must "
                f"produce pred_<stat> for all of {stats}."
            )

    frame["mew_pred"] = _mew_contribution(frame, "pred_", my_totals, gradient, group)
    frame["mew_actual"] = _mew_contribution(frame, "actual_", my_totals, gradient, group)
    frame["mew_error"] = frame["mew_pred"] - frame["mew_actual"]
    return frame


def _baseline_atc(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Unadjusted projection. The thing every candidate must beat."""
    frame = frame.copy()
    stats = HITTING_STATS if group == "hitting" else PITCHING_STATS
    for stat in stats:
        frame[f"pred_{stat}"] = frame[f"proj_{stat}"]
    return frame


def _baseline_raw_ytd(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Season-to-date rates, unshrunk, projected onto the projection's volume.

    Brown (2008) found this is worse than the league grand mean for batting
    average. It is here to confirm that finding rather than to compete.
    """
    frame = frame.copy()
    vol_col = "PA" if group == "hitting" else "IP"
    frame[f"pred_{vol_col}"] = frame[f"proj_{vol_col}"]

    if group == "hitting":
        per_pa = frame["proj_PA"] / frame["n_evid"].where(frame["n_evid"] > 0)
        for stat in ("R", "HR", "RBI", "SB", "OPS"):
            col = f"evid_{stat}"
            assert col in frame.columns, (
                f"_baseline_raw_ytd: missing {col}. assemble_backtest_frame "
                f"must carry evid_R/HR/RBI/SB/OPS for group='hitting' — this "
                f"baseline must not silently fall back to the projection."
            )
        for stat in ("R", "HR", "RBI", "SB"):
            frame[f"pred_{stat}"] = frame[f"evid_{stat}"] * per_pa
        frame["pred_OPS"] = frame["evid_OPS"]
        return frame

    # n_evid is BF for pitchers — the wrong denominator for scaling counting
    # stats by innings — so evidence-window IP (n_evid_ip) is used instead.
    per_ip = frame["proj_IP"] / frame["n_evid_ip"].where(frame["n_evid_ip"] > 0)
    for stat in ("W", "SV", "K", "ERA", "WHIP"):
        col = f"evid_{stat}"
        assert col in frame.columns, (
            f"_baseline_raw_ytd: missing {col}. assemble_backtest_frame must "
            f"carry evid_W/SV/K/ERA/WHIP for group='pitching' — this "
            f"baseline must not silently fall back to the projection."
        )
    for stat in ("W", "SV", "K"):
        frame[f"pred_{stat}"] = frame[f"evid_{stat}"] * per_ip
    for stat in ("ERA", "WHIP"):
        frame[f"pred_{stat}"] = frame[f"evid_{stat}"]
    return frame


def _baseline_flat_volume(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Projection rates, but every player gets the pool's median volume.

    Zimmerman measured ATC's preseason PA RMSE at 156 against 162 for a flat
    510 PA. If a volume model cannot beat this, it is not a model.
    """
    frame = _baseline_atc(frame, group)
    vol_col = "PA" if group == "hitting" else "IP"
    median_volume = float(frame[f"proj_{vol_col}"].median())
    scale = median_volume / frame[f"proj_{vol_col}"].where(
        frame[f"proj_{vol_col}"] > 0
    )
    counting = ("R", "HR", "RBI", "SB") if group == "hitting" else ("W", "SV", "K")
    frame[f"pred_{vol_col}"] = median_volume
    for stat in counting:
        frame[f"pred_{stat}"] = frame[f"proj_{stat}"] * scale
    return frame


BASELINES: dict = {
    "atc": _baseline_atc,
    "raw_ytd": _baseline_raw_ytd,
    "flat_volume": _baseline_flat_volume,
}


def run_baselines(
    frame: pd.DataFrame, my_totals: dict, gradient: dict, group: str = "hitting"
) -> pd.DataFrame:
    """Score every mandatory baseline on one backtest frame.

    Returns:
        One row per baseline: baseline, mae_mew, rmse_mew, and n.
    """
    rows = []
    for name, build in BASELINES.items():
        scored = score_in_mew(build(frame, group), my_totals, gradient, group)
        error = scored["mew_error"].dropna()
        rows.append(
            {
                "baseline": name,
                "mae_mew": float(error.abs().mean()),
                "rmse_mew": float(np.sqrt((error**2).mean())),
                "n": int(len(error)),
            }
        )
    result = pd.DataFrame(rows).sort_values("mae_mew").reset_index(drop=True)
    print(f"baselines ({group}):\n{result.to_string(index=False)}")
    return result
