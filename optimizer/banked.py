"""
Banked year-to-date team totals from the Fantrax standings table.

Roto standings are decided by full-season totals = banked YTD + rest-of-season.
The projection feeds are RoS-only, so the win model needs each team's banked
half as a fixed additive baseline (see lineup_solver.blend_season_totals and
MATHEMATICAL_FRAMEWORK — banked integration).

This module converts the Fantrax standings DataFrame (authoritative: it reflects
actual accrued totals regardless of mid-season roster churn) into the
``banked_totals`` dict that ``compute_league_state`` consumes.

SAFETY: ``standings_to_banked_totals`` range-validates every rate stat. If the
upstream standings parse grabbed the wrong cells (e.g. roto points instead of
category values), validation fails and the function returns None — the caller
then runs in pure rest-of-season mode rather than on corrupted banked numbers.
This makes the (hard-to-test-offline) Fantrax parsing safe to deploy.
"""

import pandas as pd

from .config import TEAM_ID_TO_NAME

# Standings column (lowercase) → category key (uppercase) used everywhere else.
_STANDINGS_COL_TO_CAT: dict[str, str] = {
    "r": "R",
    "hr": "HR",
    "rbi": "RBI",
    "sb": "SB",
    "ops": "OPS",
    "w": "W",
    "sv": "SV",
    "k": "K",
    "era": "ERA",
    "whip": "WHIP",
}

# Fantrax standings report AB, not PA. The banked OPS blend weight should be
# PA; league-wide PA/AB runs ~1.13 (PA adds walks, HBP, sacrifices). The weight
# only sets the banked-vs-ros mixing ratio, so this constant approximation is
# plenty accurate.
_PA_PER_AB: float = 1.13

# Plausible season-to-date team rate ranges. These are deliberately wide enough
# to never reject a real value, but tight enough to catch a mis-parse (e.g. a
# team OPS of 7.0 means the parser grabbed roto points, not the OPS).
_RATE_VALID_RANGES: dict[str, tuple[float, float]] = {
    "OPS": (0.300, 1.300),
    "ERA": (0.500, 12.000),
    "WHIP": (0.500, 3.000),
}


def standings_to_banked_totals(
    standings: pd.DataFrame,
) -> dict[str, dict[str, float]] | None:
    """Convert a Fantrax standings DataFrame into banked YTD totals per team.

    Args:
        standings: DataFrame from data_prep ``fetch_standings``. Must have a
            ``team_name`` column plus the 10 category-value columns
            (r, hr, rbi, sb, ops, w, sv, k, era, whip). PA/IP are NOT required
            here — the playing-time weights for ratio blending are estimated
            downstream from the season fraction (see compute_league_state).

    Returns:
        ``{team_name: {R, HR, RBI, SB, W, SV, K, OPS, ERA, WHIP, PA, IP}}`` —
        one entry per team. PA (from at-bats) and IP are the banked playing-time
        weights for ratio blending when the standings provide them; if absent
        they are omitted and the consumer estimates them.
        Returns None (→ pure rest-of-season mode) if the standings table is
        empty, missing category columns, or fails rate-range validation.
    """
    if standings is None or len(standings) == 0:
        print("banked: standings empty — using rest-of-season-only model.")
        return None

    missing_cols = [c for c in _STANDINGS_COL_TO_CAT if c not in standings.columns]
    if missing_cols or "team_name" not in standings.columns:
        print(
            f"banked: standings missing columns {missing_cols or ['team_name']} — "
            f"using rest-of-season-only model. (fetch_standings did not populate "
            f"category values.)"
        )
        return None

    # All category cells must be present and numeric.
    cat_cols = list(_STANDINGS_COL_TO_CAT)
    numeric = standings[cat_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        print(
            "banked: standings category values are missing/non-numeric — "
            "using rest-of-season-only model."
        )
        return None

    # Range-validate rate stats: catches a mis-parse before it poisons the model.
    for col, cat in _STANDINGS_COL_TO_CAT.items():
        if cat in _RATE_VALID_RANGES:
            lo, hi = _RATE_VALID_RANGES[cat]
            vals = numeric[col]
            if (vals < lo).any() or (vals > hi).any():
                bad = vals[(vals < lo) | (vals > hi)].tolist()
                print(
                    f"banked: standings {cat} values {bad} outside plausible "
                    f"range [{lo}, {hi}] — the standings parse likely grabbed the "
                    f"wrong cells. Falling back to rest-of-season-only model."
                )
                return None

    has_ab = "ab" in standings.columns
    has_ip = "ip" in standings.columns
    has_team_id = "team_id" in standings.columns

    banked: dict[str, dict[str, float]] = {}
    for _, row in standings.iterrows():
        # Key by the roster/owner name (config key), reconciled via team_id —
        # the standings display name can differ from the owner name used by the
        # rest of the pipeline. Fall back to the display name if no id match.
        team = row["team_name"]
        if has_team_id and pd.notna(row["team_id"]):
            team = TEAM_ID_TO_NAME.get(str(row["team_id"]), team)
        entry = {cat: float(row[col]) for col, cat in _STANDINGS_COL_TO_CAT.items()}
        # Real banked playing-time weights for ratio blending (PA ≈ AB × 1.13).
        if has_ab and pd.notna(row["ab"]) and float(row["ab"]) > 0:
            entry["PA"] = float(row["ab"]) * _PA_PER_AB
        if has_ip and pd.notna(row["ip"]) and float(row["ip"]) > 0:
            entry["IP"] = float(row["ip"])
        banked[team] = entry

    print(f"banked: loaded YTD totals for {len(banked)} teams from standings.")
    return banked
