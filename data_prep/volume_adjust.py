"""Correct rest-of-season playing time using in-season evidence.

Volume is the only gradient-invariant input (spec §1.4): a PA error moves
every counting category AND re-weights the ratio, so its value holds in every
league state. That is why this ships and rate correction is gated.

Model, from Zimmerman's three validated drivers plus MGL's slump-benching
effect:

    log(PA_actual / PA_proj) = b0
                             + b_age    * (age - 30)
                             + b_talent * (proj_OPS - 0.730)
                             + b_slump  * (ytd_OPS - proj_OPS)

b0 absorbs the systematic ~10% over-projection every system shows.
"""

import numpy as np
import pandas as pd

# Reference points, so coefficients read as deviations rather than intercept soup.
_AGE_REFERENCE: float = 30.0
_OPS_REFERENCE: float = 0.730

_HITTER_COUNTING: tuple[str, ...] = ("AB", "R", "HR", "RBI", "SB")
_PITCHER_COUNTING: tuple[str, ...] = ("W", "SV", "K")

REQUIRED_COEFFICIENTS: tuple[str, ...] = (
    "b0", "b_age", "b_talent", "b_slump", "min_factor", "max_factor",
)


def _volume_factor(
    frame: pd.DataFrame, coefficients: dict[str, float]
) -> pd.Series:
    """Multiplier on projected volume. 1.0 where evidence is missing."""
    log_factor = (
        coefficients["b0"]
        + coefficients["b_age"] * (frame["age"] - _AGE_REFERENCE)
        + coefficients["b_talent"] * (frame["proj_OPS"] - _OPS_REFERENCE)
        + coefficients["b_slump"] * (frame["ytd_OPS"] - frame["proj_OPS"])
    )
    factor = np.exp(log_factor)
    # No evidence -> no opinion. Never guess a player into or out of a lineup.
    factor = factor.where(frame["ytd_OPS"].notna(), 1.0)
    factor = factor.where(frame["age"].notna(), 1.0)
    return factor.clip(
        lower=coefficients["min_factor"], upper=coefficients["max_factor"]
    ).astype(float)


def adjust_projection_volume(
    players: pd.DataFrame, ytd: pd.DataFrame, coefficients: dict[str, float]
) -> pd.DataFrame:
    """Rescale rest-of-season volume, carrying counting stats with it.

    Requires columns on `players`: Name, MLBAMID, player_type, age, and all of
    PA, AB, IP, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP.
    Requires columns on `ytd`: MLBAMID, group, n_PA, n_BF, ytd_OPS.

    Adds columns: PA_adj_factor, IP_adj_factor.
    Rewrites columns: PA, AB, R, HR, RBI, SB (hitters); IP, W, SV, K (pitchers).

    Rates (OPS, ERA, WHIP) are NOT touched — that is Part 2b, which is gated.
    """
    players = players.copy()
    missing = [c for c in REQUIRED_COEFFICIENTS if c not in coefficients]
    assert not missing, (
        f"adjust_projection_volume: coefficients missing {missing}. Fit them "
        f"with fit_volume_correction against a backtest frame; do not guess."
    )
    assert coefficients["min_factor"] > 0.0, (
        f"adjust_projection_volume: min_factor is {coefficients['min_factor']}; "
        f"it must be strictly positive. Zero volume drops a player from FV's "
        f"ratio z-population and permanently benches him if he is on the IL."
    )

    evidence = ytd[["MLBAMID", "ytd_OPS"]].drop_duplicates("MLBAMID")
    frame = players[["MLBAMID", "age", "OPS"]].rename(columns={"OPS": "proj_OPS"})
    frame = frame.merge(evidence, on="MLBAMID", how="left")
    frame.index = players.index

    factor = _volume_factor(frame, coefficients)
    is_hitter = players["player_type"] == "hitter"
    is_pitcher = players["player_type"] == "pitcher"

    players["PA_adj_factor"] = factor.where(is_hitter, 1.0)
    players["IP_adj_factor"] = factor.where(is_pitcher, 1.0)

    for col in ("PA", *_HITTER_COUNTING):
        players.loc[is_hitter, col] = (
            players.loc[is_hitter, col] * players.loc[is_hitter, "PA_adj_factor"]
        )
    for col in ("IP", *_PITCHER_COUNTING):
        players.loc[is_pitcher, col] = (
            players.loc[is_pitcher, col] * players.loc[is_pitcher, "IP_adj_factor"]
        )

    n_moved = int(((factor - 1.0).abs() > 0.01).sum())
    print(
        f"volume adjustment: {n_moved} of {len(players)} players moved >1% "
        f"(median factor {float(factor.median()):.3f})"
    )
    return players


def fit_volume_correction(
    frame: pd.DataFrame, group: str = "hitting"
) -> dict[str, float]:
    """Fit the volume multiplier by OLS on log(actual / projected) volume.

    Requires columns on `frame`: proj_PA/proj_IP, actual_PA/actual_IP,
    proj_OPS, evid_OPS, age, proj_horizon_frac.

    proj_horizon_frac (from assemble_backtest_frame) is the fraction of the
    projection's full split-to-season-end horizon that the outcome window
    actually covers. A rest-of-season projection is for the FULL horizon, so
    when the observed window is shorter, proj_PA must be scaled down to
    proj_PA * proj_horizon_frac before it's a fair comparison to actual_PA —
    otherwise the fit reads the uncovered remainder of the season as players
    losing playing time, when it is only the calendar. Dividing the raw
    ratio by proj_horizon_frac is the same correction:
    actual / (proj * frac) == (actual / proj) / frac.

    Returns the coefficient dict `adjust_projection_volume` consumes.
    """
    vol = "PA" if group == "hitting" else "IP"
    usable = frame[
        (frame[f"proj_{vol}"] > 0)
        & (frame[f"actual_{vol}"] > 0)
        & frame["age"].notna()
        & frame["evid_OPS"].notna()
    ].copy()
    assert len(usable) >= 50, (
        f"fit_volume_correction: only {len(usable)} usable rows for {group}. "
        f"Fitting four coefficients on this is overfitting; widen the window "
        f"or fall back to the reconstruction route (spec §3.1)."
    )

    target = np.log(
        usable[f"actual_{vol}"]
        / (usable[f"proj_{vol}"] * usable["proj_horizon_frac"])
    )
    design = np.column_stack(
        [
            np.ones(len(usable)),
            usable["age"] - _AGE_REFERENCE,
            usable["proj_OPS"] - _OPS_REFERENCE,
            usable["evid_OPS"] - usable["proj_OPS"],
        ]
    )
    beta, *_ = np.linalg.lstsq(design, target.values, rcond=None)
    coefficients = {
        "b0": float(beta[0]),
        "b_age": float(beta[1]),
        "b_talent": float(beta[2]),
        "b_slump": float(beta[3]),
        "min_factor": 0.25,
        "max_factor": 2.0,
    }
    residual = target.values - design @ beta
    r_squared = 1.0 - residual.var() / target.values.var()
    print(
        f"fit_volume_correction ({group}, n={len(usable)}): R2={r_squared:.3f} "
        f"{ {k: round(v, 4) for k, v in coefficients.items()} }"
    )
    return coefficients
