"""
FanGraphs CSV loading: hitter and pitcher projections.

Depends on pandas only.
"""

import pandas as pd

from .names import strip_diacritics


def load_hitter_projections(filepath: str, min_ab: int = 10) -> pd.DataFrame:
    """
    Load hitter projections from FanGraphs CSV.

    Args:
        filepath: Path to hitter projections CSV
        min_ab: Minimum projected AB to include a player. Filters out
            micro-projected minor leaguers that are noise. Set to 0 to disable.

    Returns:
        DataFrame with columns:
            Name (with -H suffix), Team, Position, PA, R, HR, RBI, SB, OPS,
            player_type='hitter', WAR, MLBAMID
    """
    df = pd.read_csv(filepath)

    if min_ab > 0 and "AB" in df.columns:
        before = len(df)
        df = df[df["AB"] >= min_ab].reset_index(drop=True)
        filtered = before - len(df)
        if filtered > 0:
            print(
                f"  Filtered {filtered} hitters with AB < {min_ab} ({len(df)} remaining)"
            )

    df["Name"] = df["Name"].astype(str).apply(strip_diacritics) + "-H"
    df["Team"] = df["Team"].fillna("FA")
    df["Position"] = "DH"
    df["player_type"] = "hitter"

    duplicates = df[df["Name"].duplicated(keep="first")]["Name"].unique()
    if len(duplicates) > 0:
        print(
            f"  Note: Dropping {len(duplicates)} duplicate hitter names: {list(duplicates)[:5]}..."
        )
        df = df.drop_duplicates(subset="Name", keep="first")

    required_cols = [
        "Name",
        "Team",
        "Position",
        "PA",
        "R",
        "HR",
        "RBI",
        "SB",
        "OPS",
        "player_type",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        assert df[col].notna().all(), f"Found null values in column: {col}"

    if "WAR" in df.columns:
        df["WAR"] = df["WAR"].fillna(0.0)
    else:
        df["WAR"] = 0.0

    if "MLBAMID" not in df.columns:
        df["MLBAMID"] = None

    print(f"Loaded {len(df)} hitter projections (positions set by Fantrax merge)")

    return_cols = required_cols + ["WAR", "MLBAMID"]
    return df[return_cols].copy()


def load_pitcher_projections(filepath: str, min_ip: float = 5.0) -> pd.DataFrame:
    """
    Load pitcher projections from FanGraphs CSV.

    Args:
        filepath: Path to pitcher projections CSV
        min_ip: Minimum projected IP to include a player. Filters out
            micro-projected pitchers that are noise. Set to 0 to disable.

    Returns:
        DataFrame with columns:
            Name (with -P suffix), Team, Position, IP, W, SV, K, ERA, WHIP,
            player_type='pitcher', WAR, MLBAMID
    """
    df = pd.read_csv(filepath)

    if min_ip > 0:
        before = len(df)
        df = df[df["IP"] >= min_ip].reset_index(drop=True)
        filtered = before - len(df)
        if filtered > 0:
            print(
                f"  Filtered {filtered} pitchers with IP < {min_ip} ({len(df)} remaining)"
            )

    df["Name"] = df["Name"].astype(str).apply(strip_diacritics) + "-P"
    df["Team"] = df["Team"].fillna("FA")
    df = df.rename(columns={"SO": "K"})
    df["Position"] = "RP"
    df["player_type"] = "pitcher"

    duplicates = df[df["Name"].duplicated(keep="first")]["Name"].unique()
    if len(duplicates) > 0:
        print(
            f"  Note: Dropping {len(duplicates)} duplicate pitcher names: {list(duplicates)[:5]}..."
        )
        df = df.drop_duplicates(subset="Name", keep="first")

    required_cols = [
        "Name",
        "Team",
        "Position",
        "IP",
        "W",
        "SV",
        "K",
        "ERA",
        "WHIP",
        "player_type",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        assert df[col].notna().all(), f"Found null values in column: {col}"

    if "WAR" in df.columns:
        df["WAR"] = df["WAR"].fillna(0.0)
    else:
        df["WAR"] = 0.0

    if "MLBAMID" not in df.columns:
        df["MLBAMID"] = None

    print(f"Loaded {len(df)} pitcher projections (positions set by Fantrax merge)")

    return_cols = required_cols + ["WAR", "MLBAMID"]
    return df[return_cols].copy()
