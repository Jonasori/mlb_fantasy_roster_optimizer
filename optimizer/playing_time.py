"""
Playing time adjustment for projection systems.

All projection systems systematically overproject playing time. This module
applies a three-factor adjustment based on Jeff Zimmerman's research:
1. Historical PA/IP (strongest predictor)
2. Age (older players overprojected)
3. Talent (weaker players overprojected)

Usage:
    from optimizer.playing_time import adjust_projections
    adjust_projections()  # Uses default paths
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input paths
MLB_STATS_DB = Path("../mlb_player_comps_dashboard/mlb_stats.db")
OPTIMIZER_DB = Path("data/optimizer.db")
DEFAULT_HITTERS_INPUT = Path("data/fangraphs-atc-projections-hitters.csv")
DEFAULT_PITCHERS_INPUT = Path("data/fangraphs-atc-projections-pitchers.csv")

# Output paths (structurally identical to FanGraphs originals)
DEFAULT_HITTERS_OUTPUT = Path("data/fangraphs-atc-pt-adjusted-hitters.csv")
DEFAULT_PITCHERS_OUTPUT = Path("data/fangraphs-atc-pt-adjusted-pitchers.csv")

# Playing time baselines
FULL_SEASON_PA = 600
FULL_SEASON_IP_SP = 180
FULL_SEASON_IP_RP = 70

# Age penalty: 5% per year over threshold, floor at 50%
AGE_THRESHOLD_HITTER = 32
AGE_THRESHOLD_PITCHER = 33
AGE_PENALTY_PER_YEAR = 0.05

# Talent penalty: 15% for bottom quartile
TALENT_PERCENTILE = 0.25
TALENT_PENALTY = 0.85

# Blend: 65% projection + 35% adjusted
PROJECTION_WEIGHT = 0.65
ADJUSTMENT_WEIGHT = 0.35


# =============================================================================
# CORE ADJUSTMENT FUNCTIONS
# =============================================================================


def _compute_historical_ratio(actual_current, actual_prior, full_season):
    """Ratio of actual to expected playing time, capped at 1.0."""
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

    return min(1.0, avg / full_season)


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


def _load_historical(db_path, seasons=(2024, 2023)):
    """Load historical PA/IP from mlb_stats.db."""
    if not db_path.exists():
        return {"hitters": pd.DataFrame(), "pitchers": pd.DataFrame()}

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
    return {"hitters": hitters, "pitchers": pitchers}


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


def _adjust_hitters(hitters_df, historical_df, ages_df):
    """Adjust PA in hitters DataFrame, preserving all other columns."""
    df = hitters_df.copy()

    # Pivot historical to wide format
    if len(historical_df) > 0:
        hist = historical_df.pivot(index="mlbam_id", columns="season", values="pa")
        hist = hist.add_prefix("pa_").reset_index()
        for col in ["pa_2024", "pa_2023"]:
            if col not in hist.columns:
                hist[col] = np.nan
    else:
        hist = pd.DataFrame(columns=["mlbam_id", "pa_2024", "pa_2023"])

    # Join
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

    # Compute adjustment
    war_25 = df["WAR"].quantile(TALENT_PERCENTILE)
    for idx, row in df.iterrows():
        hist_ratio = _compute_historical_ratio(
            row.get("pa_2024"), row.get("pa_2023"), FULL_SEASON_PA
        )
        age_factor = _compute_age_factor(row.get("age"), is_pitcher=False)
        talent_factor = _compute_talent_factor(row["WAR"], war_25)
        df.at[idx, "PA"] = _blend(row["PA"], hist_ratio, age_factor, talent_factor)

    # Clean up temporary columns
    drop_cols = [c for c in df.columns if c.startswith(("pa_2", "mlbam_id"))] + ["age"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def _adjust_pitchers(pitchers_df, historical_df, ages_df):
    """Adjust IP in pitchers DataFrame, preserving all other columns."""
    df = pitchers_df.copy()

    # Pivot historical to wide format
    if len(historical_df) > 0:
        hist_ip = historical_df.pivot(
            index="mlbam_id", columns="season", values="ip"
        ).add_prefix("ip_")
        hist_gs = historical_df.pivot(
            index="mlbam_id", columns="season", values="gs"
        ).add_prefix("gs_")
        hist = hist_ip.join(hist_gs, how="outer").reset_index()
        for col in ["ip_2024", "ip_2023", "gs_2024", "gs_2023"]:
            if col not in hist.columns:
                hist[col] = np.nan
    else:
        hist = pd.DataFrame(columns=["mlbam_id", "ip_2024", "ip_2023"])

    # Join
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

    # Compute adjustment
    war_25 = df["WAR"].quantile(TALENT_PERCENTILE)
    for idx, row in df.iterrows():
        is_sp = row.get("GS", 0) >= 3
        full_ip = FULL_SEASON_IP_SP if is_sp else FULL_SEASON_IP_RP
        hist_ratio = _compute_historical_ratio(
            row.get("ip_2024"), row.get("ip_2023"), full_ip
        )
        age_factor = _compute_age_factor(row.get("age"), is_pitcher=True)
        talent_factor = _compute_talent_factor(row["WAR"], war_25)
        df.at[idx, "IP"] = _blend(row["IP"], hist_ratio, age_factor, talent_factor)

    # Clean up temporary columns
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
) -> dict:
    """
    Adjust ATC projections for playing time bias.

    Reads the input CSVs, applies the three-factor adjustment (historical,
    age, talent), and writes output CSVs that are structurally identical
    to the originals with only PA/IP values changed.

    Args:
        hitters_input: Path to FanGraphs hitters CSV
        pitchers_input: Path to FanGraphs pitchers CSV
        hitters_output: Path for adjusted hitters output
        pitchers_output: Path for adjusted pitchers output
        mlb_stats_db: Path to historical stats database
        optimizer_db: Path to optimizer database (for ages)

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

    # Load input projections
    hitters = pd.read_csv(hitters_input)
    pitchers = pd.read_csv(pitchers_input)
    orig_pa = hitters["PA"].sum()
    orig_ip = pitchers["IP"].sum()

    # Load historical data and ages
    historical = _load_historical(mlb_stats_db)
    ages = _load_ages(optimizer_db)

    # Apply adjustments
    adj_hitters = _adjust_hitters(hitters, historical["hitters"], ages)
    adj_pitchers = _adjust_pitchers(pitchers, historical["pitchers"], ages)

    # Write output
    hitters_output.parent.mkdir(parents=True, exist_ok=True)
    adj_hitters.to_csv(hitters_output, index=False)
    adj_pitchers.to_csv(pitchers_output, index=False)

    # Summary
    adj_pa = adj_hitters["PA"].sum()
    adj_ip = adj_pitchers["IP"].sum()
    pa_reduction = 1 - adj_pa / orig_pa
    ip_reduction = 1 - adj_ip / orig_ip

    print(f"  Hitters: {len(adj_hitters)}, PA reduced by {pa_reduction:.1%}")
    print(f"  Pitchers: {len(adj_pitchers)}, IP reduced by {ip_reduction:.1%}")
    print(f"  Output: {hitters_output}, {pitchers_output}")

    return {
        "hitters": len(adj_hitters),
        "pitchers": len(adj_pitchers),
        "pa_reduction": pa_reduction,
        "ip_reduction": ip_reduction,
    }
