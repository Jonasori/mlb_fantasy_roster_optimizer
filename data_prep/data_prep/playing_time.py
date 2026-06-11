"""
Playing time adjustment for projection systems.

WARNING: This module is designed for PRESEASON FULL-SEASON projections only.
It is NOT part of the default pipeline anymore and must NOT be run on
rest-of-season (RoS) feeds: it compares prior full-season historical PA/IP to
the projected PA/IP denominator, which is meaningless (and inflating) when the
projection is remaining-season only. RoS feeds already reflect current playing
time, so no adjustment is applied.

All projection systems systematically overproject playing time. This module
applies a three-factor adjustment based on Jeff Zimmerman's research:
1. Historical PA/IP (strongest predictor)
2. Age (older players overprojected)
3. Talent (weaker players overprojected)

When PA/IP is adjusted, counting stats (HR, R, RBI, SB, W, SV, K, etc.)
are proportionally rescaled. FanGraphs projections are pre-scaled totals
(counting_stat = rate × playing_time), so adjusting playing time without
rescaling counting stats would create an internal inconsistency. Rate stats
(OPS, ERA, WHIP, AVG, etc.) are unaffected — they don't depend on volume.

Usage:
    from data_prep.playing_time import adjust_projections
    adjust_projections()  # Uses default paths under data_prep/data/
"""

import datetime
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# data_prep project root (contains pyproject.toml and data/)
_DATA_PREP_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _DATA_PREP_ROOT / "data"
# Repo root (parent of data_prep package directory)
_REPO_ROOT = _DATA_PREP_ROOT.parent

# External historical stats DB (sibling project to repo by default)
MLB_STATS_DB = _REPO_ROOT.parent / "mlb_player_comps_dashboard" / "mlb_stats.db"
OPTIMIZER_DB = _DATA_DIR / "optimizer.db"
DEFAULT_HITTERS_INPUT = _DATA_DIR / "fangraphs-atc-projections-hitters_ros.csv"
DEFAULT_PITCHERS_INPUT = _DATA_DIR / "fangraphs-atc-projections-pitchers_ros.csv"

# Output paths (structurally identical to FanGraphs originals)
DEFAULT_HITTERS_OUTPUT = _DATA_DIR / "fangraphs-atc-pt-adjusted-hitters.csv"
DEFAULT_PITCHERS_OUTPUT = _DATA_DIR / "fangraphs-atc-pt-adjusted-pitchers.csv"

# Age penalty: 5% per year over threshold, floor at 50%
AGE_THRESHOLD_HITTER = 32
AGE_THRESHOLD_PITCHER = 33
AGE_PENALTY_PER_YEAR = 0.05

# Talent penalty: 15% for bottom quartile (uses WAR rate, not raw WAR,
# to avoid conflating low playing time with low talent)
TALENT_PERCENTILE = 0.25
TALENT_PENALTY = 0.85

# Blend: 65% projection + 35% adjusted
# Bayesian interpretation: projection is ~1.86x more precise than
# 2-year historical average for predicting next-year PA/IP.
PROJECTION_WEIGHT = 0.65
ADJUSTMENT_WEIGHT = 0.35

# Historical ratio cap: allows modest upward adjustments (up to 15%)
# for players whose recent PA/IP exceeds the projection.
HISTORICAL_RATIO_CAP = 1.15

# Hitter counting stats that scale proportionally with PA.
# FanGraphs projections store these as pre-scaled totals (rate * PA).
# Rate stats (AVG, OBP, SLG, OPS, wOBA, ISO, BABIP, wRC+, BB%, K%, etc.)
# are NOT scaled — they are independent of playing time volume.
HITTER_COUNTING_COLS = [
    "G",
    "AB",
    "H",
    "1B",
    "2B",
    "3B",
    "HR",
    "R",
    "RBI",
    "BB",
    "IBB",
    "SO",
    "HBP",
    "SF",
    "SH",
    "GDP",
    "SB",
    "CS",
    "UBR",
    "wSB",
    "wRC",
    "wRAA",
    "BsR",
    "Fld",
    "Off",
    "Def",
    "WAR",
    "FPTS",
    "SPTS",
]

# Pitcher counting stats that scale proportionally with IP.
# Rate stats (ERA, WHIP, K/9, BB/9, FIP, BABIP, LOB%, GB%, HR/FB, etc.)
# are NOT scaled.
PITCHER_COUNTING_COLS = [
    "W",
    "L",
    "QS",
    "G",
    "GS",
    "SV",
    "HLD",
    "BS",
    "TBF",
    "H",
    "R",
    "ER",
    "HR",
    "BB",
    "IBB",
    "HBP",
    "SO",
    "WAR",
    "RA9-WAR",
    "FPTS",
    "SPTS",
]


# =============================================================================
# CORE ADJUSTMENT FUNCTIONS
# =============================================================================


def _compute_historical_ratio(actual_current, actual_prior, projected):
    """Ratio of actual playing time to projected, capped at HISTORICAL_RATIO_CAP.

    Uses the player's own projected PA/IP as the denominator so that
    part-time players aren't double-penalized (once by a low projection,
    again by comparison to a full-season baseline).

    Allows modest upward adjustments (up to HISTORICAL_RATIO_CAP) for
    players whose recent playing time exceeds the projection.
    """
    if projected <= 0:
        return 1.0

    has_current = actual_current is not None and not pd.isna(actual_current)
    has_prior = actual_prior is not None and not pd.isna(actual_prior)

    if has_current and has_prior:
        avg = (actual_current + actual_prior) / 2
    elif has_current:
        avg = actual_current
    elif has_prior:
        avg = actual_prior
    else:
        return 1.0  # Trust projection for rookies

    return min(HISTORICAL_RATIO_CAP, avg / projected)


def _compute_age_factor(age, is_pitcher):
    """Age-based reduction, 5% per year over threshold."""
    if age is None or pd.isna(age):
        return 1.0

    threshold = AGE_THRESHOLD_PITCHER if is_pitcher else AGE_THRESHOLD_HITTER
    if age < threshold:
        return 1.0

    years_over = age - threshold + 1
    return max(0.5, 1.0 - AGE_PENALTY_PER_YEAR * years_over)


def _compute_talent_factor(war, war_25th):
    """15% penalty for bottom quartile."""
    if pd.isna(war) or pd.isna(war_25th):
        return 1.0
    return TALENT_PENALTY if war < war_25th else 1.0


def _blend(projected, historical_ratio, age_factor, talent_factor):
    """Apply factors and blend with original projection."""
    adjusted = projected * historical_ratio * age_factor * talent_factor
    return PROJECTION_WEIGHT * projected + ADJUSTMENT_WEIGHT * adjusted


# =============================================================================
# DATA LOADING
# =============================================================================


def _default_historical_seasons() -> tuple[int, int]:
    """Two most recent completed MLB seasons."""
    current_year = datetime.date.today().year
    return (current_year - 1, current_year - 2)


def _load_historical(db_path, seasons=None):
    """Load historical PA/IP from mlb_stats.db.

    Args:
        db_path: Path to the mlb_stats.db file.
        seasons: Tuple of season years to load. Defaults to the two most
            recent completed seasons based on the current date.
    """
    if seasons is None:
        seasons = _default_historical_seasons()

    if not db_path.exists():
        return {
            "hitters": pd.DataFrame(),
            "pitchers": pd.DataFrame(),
            "seasons": seasons,
        }

    seasons_str = ",".join(str(s) for s in seasons)
    conn = sqlite3.connect(db_path)

    hitters = pd.read_sql(
        f"""
        SELECT p.player_id as mlbam_id, g.season, SUM(g.pa) as pa
        FROM players p JOIN game_logs g ON p.player_id = g.player_id
        WHERE g.season IN ({seasons_str})
        GROUP BY p.player_id, g.season
        """,
        conn,
    )

    pitchers = pd.read_sql(
        f"""
        SELECT p.player_id as mlbam_id, g.season, SUM(g.ip) as ip, SUM(g.gs) as gs
        FROM pitchers p JOIN pitcher_game_logs g ON p.player_id = g.player_id
        WHERE g.season IN ({seasons_str})
        GROUP BY p.player_id, g.season
        """,
        conn,
    )

    conn.close()
    return {"hitters": hitters, "pitchers": pitchers, "seasons": seasons}


def _load_ages(db_path):
    """Load ages from optimizer.db."""
    if not db_path.exists():
        return pd.DataFrame(columns=["mlbam_id", "age"])

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(players)")
    columns = [row[1].lower() for row in cursor.fetchall()]

    if "mlbamid" not in columns:
        conn.close()
        return pd.DataFrame(columns=["mlbam_id", "age"])

    df = pd.read_sql(
        "SELECT mlbamid as mlbam_id, age FROM players WHERE age IS NOT NULL AND mlbamid IS NOT NULL",
        conn,
    )
    conn.close()

    return df.drop_duplicates(subset=["mlbam_id"], keep="first")


# =============================================================================
# ADJUSTMENT PIPELINE
# =============================================================================


def _adjust_hitters(hitters_df, historical_df, ages_df, seasons):
    """Adjust PA and counting stats in hitters DataFrame (vectorized).

    PA is adjusted via the three-factor model (historical, age, talent).
    Counting stats (HR, R, RBI, SB, etc.) are then scaled proportionally
    to the PA adjustment so that per-PA rates remain unchanged.
    Rate stats (OPS, AVG, OBP, etc.) are not modified.

    The talent factor uses WAR/PA (rate) instead of raw WAR to avoid
    penalizing low-playing-time players whose rate-level talent is fine.
    """
    df = hitters_df.copy()

    recent_col = f"pa_{seasons[0]}"
    prior_col = f"pa_{seasons[1]}"

    if len(historical_df) > 0:
        hist = historical_df.pivot(index="mlbam_id", columns="season", values="pa")
        hist = hist.add_prefix("pa_").reset_index()
        for col in [recent_col, prior_col]:
            if col not in hist.columns:
                hist[col] = np.nan
    else:
        hist = pd.DataFrame(columns=["mlbam_id", recent_col, prior_col])

    df = df.merge(hist, left_on="MLBAMID", right_on="mlbam_id", how="left")
    if len(ages_df) > 0:
        df = df.merge(
            ages_df,
            left_on="MLBAMID",
            right_on="mlbam_id",
            how="left",
            suffixes=("", "_age"),
        )
    else:
        df["age"] = np.nan

    original_pa = df["PA"].values.copy()

    # --- Historical ratio (vectorized) ---
    recent = (
        df[recent_col].values.astype(float)
        if recent_col in df.columns
        else np.full(len(df), np.nan)
    )
    prior = (
        df[prior_col].values.astype(float)
        if prior_col in df.columns
        else np.full(len(df), np.nan)
    )
    has_recent = ~np.isnan(recent)
    has_prior = ~np.isnan(prior)
    avg_actual = np.where(
        has_recent & has_prior,
        (recent + prior) / 2,
        np.where(has_recent, recent, np.where(has_prior, prior, np.nan)),
    )
    hist_ratio = np.where(
        (original_pa > 0) & ~np.isnan(avg_actual),
        np.clip(avg_actual / original_pa, 0, HISTORICAL_RATIO_CAP),
        1.0,
    )

    # --- Age factor (vectorized) ---
    age_vals = df["age"].values.astype(float)
    years_over = np.where(
        ~np.isnan(age_vals) & (age_vals >= AGE_THRESHOLD_HITTER),
        age_vals - AGE_THRESHOLD_HITTER + 1,
        0,
    )
    age_factor = np.clip(1.0 - AGE_PENALTY_PER_YEAR * years_over, 0.5, 1.0)

    # --- Talent factor (uses WAR/PA rate to avoid PA-WAR circularity) ---
    war_rate = df["WAR"].values / np.maximum(original_pa, 1)
    valid_war = ~np.isnan(df["WAR"].values)
    war_rate_25 = (
        np.nanpercentile(war_rate[valid_war], TALENT_PERCENTILE * 100)
        if valid_war.any()
        else 0.0
    )
    talent_factor = np.where(valid_war & (war_rate < war_rate_25), TALENT_PENALTY, 1.0)

    # --- Blend and scale ---
    adjusted_pa = original_pa * hist_ratio * age_factor * talent_factor
    df["PA"] = PROJECTION_WEIGHT * original_pa + ADJUSTMENT_WEIGHT * adjusted_pa

    pa_ratio = np.where(original_pa > 0, df["PA"].values / original_pa, 1.0)
    for col in HITTER_COUNTING_COLS:
        if col in df.columns:
            df[col] = df[col].values * pa_ratio

    drop_cols = [
        c for c in df.columns if c.startswith(("pa_", "mlbam_id")) and c != "PA"
    ] + ["age"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def _adjust_pitchers(pitchers_df, historical_df, ages_df, seasons):
    """Adjust IP and counting stats in pitchers DataFrame (vectorized).

    IP is adjusted via the three-factor model (historical, age, talent).
    Counting stats (W, SV, K, etc.) are then scaled proportionally
    to the IP adjustment so that per-IP rates remain unchanged.
    Rate stats (ERA, WHIP, FIP, K/9, etc.) are not modified.

    The talent factor uses WAR/IP (rate) instead of raw WAR to avoid
    penalizing low-IP pitchers whose rate-level talent is fine.
    """
    df = pitchers_df.copy()

    recent_ip_col = f"ip_{seasons[0]}"
    prior_ip_col = f"ip_{seasons[1]}"

    if len(historical_df) > 0:
        hist_ip = historical_df.pivot(
            index="mlbam_id", columns="season", values="ip"
        ).add_prefix("ip_")
        hist_gs = historical_df.pivot(
            index="mlbam_id", columns="season", values="gs"
        ).add_prefix("gs_")
        hist = hist_ip.join(hist_gs, how="outer").reset_index()
        for col in [recent_ip_col, prior_ip_col]:
            if col not in hist.columns:
                hist[col] = np.nan
    else:
        hist = pd.DataFrame(columns=["mlbam_id", recent_ip_col, prior_ip_col])

    df = df.merge(hist, left_on="MLBAMID", right_on="mlbam_id", how="left")
    if len(ages_df) > 0:
        df = df.merge(
            ages_df,
            left_on="MLBAMID",
            right_on="mlbam_id",
            how="left",
            suffixes=("", "_age"),
        )
    else:
        df["age"] = np.nan

    original_ip = df["IP"].values.copy()

    # --- Historical ratio (vectorized) ---
    recent = (
        df[recent_ip_col].values.astype(float)
        if recent_ip_col in df.columns
        else np.full(len(df), np.nan)
    )
    prior = (
        df[prior_ip_col].values.astype(float)
        if prior_ip_col in df.columns
        else np.full(len(df), np.nan)
    )
    has_recent = ~np.isnan(recent)
    has_prior = ~np.isnan(prior)
    avg_actual = np.where(
        has_recent & has_prior,
        (recent + prior) / 2,
        np.where(has_recent, recent, np.where(has_prior, prior, np.nan)),
    )
    hist_ratio = np.where(
        (original_ip > 0) & ~np.isnan(avg_actual),
        np.clip(avg_actual / original_ip, 0, HISTORICAL_RATIO_CAP),
        1.0,
    )

    # --- Age factor (vectorized) ---
    age_vals = df["age"].values.astype(float)
    years_over = np.where(
        ~np.isnan(age_vals) & (age_vals >= AGE_THRESHOLD_PITCHER),
        age_vals - AGE_THRESHOLD_PITCHER + 1,
        0,
    )
    age_factor = np.clip(1.0 - AGE_PENALTY_PER_YEAR * years_over, 0.5, 1.0)

    # --- Talent factor (uses WAR/IP rate to avoid IP-WAR circularity) ---
    war_rate = df["WAR"].values / np.maximum(original_ip, 1)
    valid_war = ~np.isnan(df["WAR"].values)
    war_rate_25 = (
        np.nanpercentile(war_rate[valid_war], TALENT_PERCENTILE * 100)
        if valid_war.any()
        else 0.0
    )
    talent_factor = np.where(valid_war & (war_rate < war_rate_25), TALENT_PENALTY, 1.0)

    # --- Blend and scale ---
    adjusted_ip = original_ip * hist_ratio * age_factor * talent_factor
    df["IP"] = PROJECTION_WEIGHT * original_ip + ADJUSTMENT_WEIGHT * adjusted_ip

    ip_ratio = np.where(original_ip > 0, df["IP"].values / original_ip, 1.0)
    for col in PITCHER_COUNTING_COLS:
        if col in df.columns:
            df[col] = df[col].values * ip_ratio

    drop_cols = [c for c in df.columns if c.startswith(("ip_", "gs_", "mlbam_id"))] + [
        "age"
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


# =============================================================================
# PUBLIC API
# =============================================================================


def adjust_projections(
    hitters_input: str | Path = DEFAULT_HITTERS_INPUT,
    pitchers_input: str | Path = DEFAULT_PITCHERS_INPUT,
    hitters_output: str | Path = DEFAULT_HITTERS_OUTPUT,
    pitchers_output: str | Path = DEFAULT_PITCHERS_OUTPUT,
    mlb_stats_db: str | Path = MLB_STATS_DB,
    optimizer_db: str | Path = OPTIMIZER_DB,
    seasons: tuple[int, int] | None = None,
) -> dict:
    """
    Adjust projections for playing time bias.

    Reads the input CSVs, applies the three-factor adjustment (historical,
    age, talent), and writes output CSVs. Both PA/IP and counting stats
    are adjusted proportionally so that per-PA/IP rates are preserved.
    Rate stats (OPS, ERA, WHIP, etc.) are unchanged.

    Args:
        hitters_input: Path to FanGraphs hitters CSV
        pitchers_input: Path to FanGraphs pitchers CSV
        hitters_output: Path for adjusted hitters output
        pitchers_output: Path for adjusted pitchers output
        mlb_stats_db: Path to historical stats database
        optimizer_db: Path to optimizer database (for ages)
        seasons: Two most recent completed seasons for historical lookup.
            Defaults to (current_year - 1, current_year - 2).

    Returns:
        Dict with summary stats: {
            "hitters": int,
            "pitchers": int,
            "pa_reduction": float,
            "ip_reduction": float,
        }
    """
    hitters_input = Path(hitters_input)
    pitchers_input = Path(pitchers_input)
    hitters_output = Path(hitters_output)
    pitchers_output = Path(pitchers_output)
    mlb_stats_db = Path(mlb_stats_db)
    optimizer_db = Path(optimizer_db)

    print("Playing time adjustment...")

    hitters = pd.read_csv(hitters_input)
    pitchers = pd.read_csv(pitchers_input)
    orig_pa = hitters["PA"].sum()
    orig_ip = pitchers["IP"].sum()

    historical = _load_historical(mlb_stats_db, seasons=seasons)
    ages = _load_ages(optimizer_db)
    used_seasons = historical["seasons"]

    print(f"  Historical seasons: {used_seasons[0]}, {used_seasons[1]}")

    adj_hitters = _adjust_hitters(hitters, historical["hitters"], ages, used_seasons)
    adj_pitchers = _adjust_pitchers(
        pitchers, historical["pitchers"], ages, used_seasons
    )

    hitters_output.parent.mkdir(parents=True, exist_ok=True)
    adj_hitters.to_csv(hitters_output, index=False)
    adj_pitchers.to_csv(pitchers_output, index=False)

    adj_pa = adj_hitters["PA"].sum()
    adj_ip = adj_pitchers["IP"].sum()
    pa_reduction = 1 - adj_pa / orig_pa
    ip_reduction = 1 - adj_ip / orig_ip

    print(
        f"  Hitters: {len(adj_hitters)}, PA reduced by {pa_reduction:.1%} (counting stats scaled proportionally)"
    )
    print(
        f"  Pitchers: {len(adj_pitchers)}, IP reduced by {ip_reduction:.1%} (counting stats scaled proportionally)"
    )
    print(f"  Output: {hitters_output}, {pitchers_output}")

    return {
        "hitters": len(adj_hitters),
        "pitchers": len(adj_pitchers),
        "pa_reduction": pa_reduction,
        "ip_reduction": ip_reduction,
    }
