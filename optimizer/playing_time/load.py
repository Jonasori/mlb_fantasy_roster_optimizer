"""Load data from external sources."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ATC_HITTERS, ATC_PITCHERS, MLB_STATS_DB, OPTIMIZER_DB


def load_atc_hitters(filepath: Path) -> pd.DataFrame:
    """
    Load ATC hitter projections.

    Returns:
        DataFrame with columns: Name, MLBAMID, Team, PA, R, HR, RBI, SB, OPS, WAR
        Plus player_type = 'hitter'
    """
    df = pd.read_csv(filepath)

    # Select required columns
    required_cols = [
        "Name",
        "MLBAMID",
        "Team",
        "PA",
        "R",
        "HR",
        "RBI",
        "SB",
        "OPS",
        "WAR",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column in hitters CSV: {col}"

    df = df[required_cols].copy()

    # Add player_type
    df["player_type"] = "hitter"

    # Validate
    assert df["MLBAMID"].notna().all(), "All hitters must have MLBAMID"
    assert (df["PA"] > 0).all(), "All hitters must have PA > 0"

    return df


def load_atc_pitchers(filepath: Path) -> pd.DataFrame:
    """
    Load ATC pitcher projections.

    Returns:
        DataFrame with columns: Name, MLBAMID, Team, IP, W, SV, K, ERA, WHIP, GS, WAR
        Plus player_type = 'pitcher', Position = 'SP' or 'RP'
    """
    df = pd.read_csv(filepath)

    # Rename SO -> K
    df = df.rename(columns={"SO": "K"})

    # Select required columns
    required_cols = [
        "Name",
        "MLBAMID",
        "Team",
        "IP",
        "W",
        "SV",
        "K",
        "ERA",
        "WHIP",
        "GS",
        "WAR",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column in pitchers CSV: {col}"

    df = df[required_cols].copy()

    # Add Position = 'SP' if GS >= 3 else 'RP'
    df["Position"] = np.where(df["GS"] >= 3, "SP", "RP")

    # Add player_type
    df["player_type"] = "pitcher"

    # Validate
    assert df["MLBAMID"].notna().all(), "All pitchers must have MLBAMID"
    assert (df["IP"] > 0).all(), "All pitchers must have IP > 0"

    return df


def load_atc_projections(
    hitters_path: Path = ATC_HITTERS,
    pitchers_path: Path = ATC_PITCHERS,
) -> pd.DataFrame:
    """
    Load and combine ATC hitter and pitcher projections.

    Returns:
        Combined DataFrame with all columns aligned.
        Hitters have IP=0, Pitchers have PA=0.
    """
    hitters = load_atc_hitters(hitters_path)
    pitchers = load_atc_pitchers(pitchers_path)

    # Add missing columns with appropriate defaults
    # Hitters don't have pitching stats
    hitters["IP"] = 0.0
    hitters["W"] = 0
    hitters["SV"] = 0
    hitters["K"] = 0
    hitters["ERA"] = 0.0
    hitters["WHIP"] = 0.0
    hitters["GS"] = 0
    hitters["Position"] = "UTIL"  # Default position for hitters

    # Pitchers don't have hitting stats
    pitchers["PA"] = 0
    pitchers["R"] = 0
    pitchers["HR"] = 0
    pitchers["RBI"] = 0
    pitchers["SB"] = 0
    pitchers["OPS"] = 0.0

    # Align column order
    all_cols = [
        "Name",
        "MLBAMID",
        "Team",
        "PA",
        "R",
        "HR",
        "RBI",
        "SB",
        "OPS",
        "IP",
        "W",
        "SV",
        "K",
        "ERA",
        "WHIP",
        "GS",
        "WAR",
        "player_type",
        "Position",
    ]

    hitters = hitters[all_cols]
    pitchers = pitchers[all_cols]

    # Combine
    combined = pd.concat([hitters, pitchers], ignore_index=True)

    print(f"Loaded ATC projections: {len(hitters)} hitters, {len(pitchers)} pitchers")

    return combined


def load_historical_actuals(
    db_path: Path = MLB_STATS_DB,
    seasons: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load historical PA/IP from mlb_stats.db.

    Args:
        db_path: Path to mlb_stats.db
        seasons: List of seasons to load (default: [2023, 2024])

    Returns:
        {
            "hitters": DataFrame [mlbam_id, season, pa],
            "pitchers": DataFrame [mlbam_id, season, ip, gs]
        }
    """
    if seasons is None:
        seasons = [2023, 2024]

    seasons_str = ",".join(str(s) for s in seasons)

    conn = sqlite3.connect(db_path)

    # Query hitter season totals
    hitters_query = f"""
        SELECT 
            p.player_id as mlbam_id,
            g.season,
            SUM(g.pa) as pa
        FROM players p
        JOIN game_logs g ON p.player_id = g.player_id
        WHERE g.season IN ({seasons_str})
        GROUP BY p.player_id, g.season
    """
    hitters_df = pd.read_sql(hitters_query, conn)

    # Query pitcher season totals
    pitchers_query = f"""
        SELECT 
            p.player_id as mlbam_id,
            g.season,
            SUM(g.ip) as ip,
            SUM(g.gs) as gs
        FROM pitchers p
        JOIN pitcher_game_logs g ON p.player_id = g.player_id
        WHERE g.season IN ({seasons_str})
        GROUP BY p.player_id, g.season
    """
    pitchers_df = pd.read_sql(pitchers_query, conn)

    conn.close()

    print(
        f"Loaded historical: {len(hitters_df)} hitter-seasons, {len(pitchers_df)} pitcher-seasons"
    )

    return {"hitters": hitters_df, "pitchers": pitchers_df}


def load_ages(db_path: Path = OPTIMIZER_DB) -> pd.DataFrame:
    """
    Load player ages from optimizer database.

    Args:
        db_path: Path to data/optimizer.db

    Returns:
        DataFrame with columns [mlbam_id, age]
        Only players with non-null age.

    Note:
        Ages are populated by the data pipeline via Fantrax player pool.
        If database doesn't exist or has no ages, returns empty DataFrame
        and the adjustment will skip the age factor.
    """
    # Check if database exists
    if not db_path.exists():
        print("WARNING: No ages found in database. Age adjustment will be skipped.")
        return pd.DataFrame(columns=["mlbam_id", "age"])

    conn = sqlite3.connect(db_path)

    # The optimizer.db players table uses 'name' as primary key, not MLBAMID
    # We need to check if there's a way to get ages with MLBAM IDs
    # First, check the schema
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(players)")
    columns = [row[1] for row in cursor.fetchall()]

    # Check if mlbamid column exists (it may be named differently)
    mlbam_col = None
    for col in columns:
        if col.lower() in ("mlbamid", "mlbam_id"):
            mlbam_col = col
            break

    if mlbam_col is None:
        # No MLBAMID column - we need to match by name instead
        # Return empty and handle gracefully
        conn.close()
        print("WARNING: No MLBAMID column in database. Age adjustment will be skipped.")
        return pd.DataFrame(columns=["mlbam_id", "age"])

    query = f"""
        SELECT {mlbam_col} as mlbam_id, age 
        FROM players 
        WHERE age IS NOT NULL AND {mlbam_col} IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) == 0:
        print("WARNING: No ages found in database. Age adjustment will be skipped.")
    else:
        print(f"Loaded ages for {len(df)} players from database")

    return df
