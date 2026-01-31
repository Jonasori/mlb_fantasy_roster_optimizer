"""
Data loading and configuration for MLB Fantasy Roster Optimizer.

This module handles:
- League configuration constants
- Loading FanGraphs projections
- Loading positions from SQLite database
- Converting Fantrax roster exports
- Name matching and correction
- Computing team totals
"""

import sqlite3
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Import all configuration constants from config module
from .config import (
    ALL_CATEGORIES,
    BALANCE_LAMBDA_DEFAULT,
    FANTRAX_ACTIVE_STATUS_IDS,
    FANTRAX_LEAGUE_ID,
    FANTRAX_TEAM_IDS,
    HITTING_CATEGORIES,
    HITTING_SLOTS,
    MAX_HITTERS,
    MAX_PITCHERS,
    MIN_HITTERS,
    MIN_PITCHERS,
    MIN_STAT_STANDARD_DEVIATION,
    MY_TEAM_ID,
    MY_TEAM_NAME,
    NEGATIVE_CATEGORIES,
    NUM_OPPONENTS,
    PITCHING_CATEGORIES,
    PITCHING_SLOTS,
    RATIO_STATS,
    ROSTER_SIZE,
    SGP_DENOMINATORS,
    SGP_RATE_STATS,
    SLOT_ELIGIBILITY,
)


# === UTILITY FUNCTIONS ===


def strip_name_suffix(name: str) -> str:
    """Strip -H or -P suffix from player name for display."""
    if name.endswith("-H") or name.endswith("-P"):
        return name[:-2]
    return name


def compute_sgp_value(row: pd.Series) -> float:
    """
    Compute Standing Gain Points (SGP) value for a single player.

    SGP is a context-free player valuation metric that estimates how many
    points in the standings a player contributes across all scoring categories.
    Higher SGP = more valuable player.

    Args:
        row: A pandas Series containing player stats (from projections DataFrame).
            Must include: player_type, R, HR, RBI, SB, OPS (for hitters)
                         or W, SV, K, ERA, WHIP (for pitchers)

    Returns:
        Total SGP value (float). Typical range: 0 to 25 for hitters,
        0 to 15 for pitchers.

    Note:
        For rate stats (OPS, ERA, WHIP), contribution is measured relative to
        league average WITHOUT playing time weighting. Playing time value is
        already captured in the counting stats. For "lower is better" stats
        (ERA, WHIP), the sign is flipped so positive SGP = better.
    """
    sgp = 0.0

    if row["player_type"] == "hitter":
        # Counting stats: R, HR, RBI, SB
        for stat in ["R", "HR", "RBI", "SB"]:
            sgp += row[stat] / SGP_DENOMINATORS[stat]

        # OPS (rate stat, higher is better, no PA weighting)
        denom, league_avg, _ = SGP_RATE_STATS["OPS"]
        sgp += (row["OPS"] - league_avg) / denom

    else:  # pitcher
        # Counting stats: W, SV, K
        for stat in ["W", "SV", "K"]:
            sgp += row[stat] / SGP_DENOMINATORS[stat]

        # ERA (rate stat, lower is better, no IP weighting)
        denom, league_avg, _ = SGP_RATE_STATS["ERA"]
        # Flip sign: (league_avg - player_ERA) so lower ERA = positive SGP
        sgp += (league_avg - row["ERA"]) / denom

        # WHIP (rate stat, lower is better, no IP weighting)
        denom, league_avg, _ = SGP_RATE_STATS["WHIP"]
        sgp += (league_avg - row["WHIP"]) / denom

    return sgp


# === POSITION LOADING ===


def load_positions_from_db(db_path: str) -> dict[int, str]:
    """
    Load player positions from SQLite database.

    Args:
        db_path: Path to optimizer.db

    Returns:
        Dict mapping MLBAMID (int) to position (str).
        Example: {592450: 'OF', 677951: 'SS', ...}

    Note:
        If database is empty or positions don't exist yet, returns empty dict.
        Positions will default to "DH" for hitters in load_hitter_projections().
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if players table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='players'"
    )
    if not cursor.fetchone():
        conn.close()
        print("  No players table found - positions will default to 'DH'")
        return {}

    # Query mlbamid and position from optimizer.db
    cursor.execute("SELECT mlbamid, position FROM players WHERE mlbamid IS NOT NULL")
    rows = cursor.fetchall()
    positions = {int(row[0]): row[1] for row in rows if row[0] is not None}
    conn.close()

    if len(positions) > 0:
        print(f"Loaded positions for {len(positions)} players from database")
    else:
        print("  No positions found in database - positions will default to 'DH'")

    return positions


# === PROJECTION LOADING ===


def load_hitter_projections(filepath: str, positions: dict[int, str]) -> pd.DataFrame:
    """
    Load hitter projections from FanGraphs CSV.

    Args:
        filepath: Path to hitter projections CSV
        positions: Dict mapping MLBAMID to position

    Returns:
        DataFrame with columns:
            Name (with -H suffix), Team, Position, PA, R, HR, RBI, SB, OPS,
            player_type='hitter', WAR, MLBAMID
    """
    df = pd.read_csv(filepath)

    # Append -H suffix to all names
    df["Name"] = df["Name"].astype(str) + "-H"

    # Fill null Team with 'FA' (free agent)
    df["Team"] = df["Team"].fillna("FA")

    # Join position from positions dict using MLBAMID
    df["Position"] = df["MLBAMID"].map(positions).fillna("DH")

    n_with_pos = df["Position"].ne("DH").sum()
    n_dh_default = len(df) - n_with_pos

    # Add player_type
    df["player_type"] = "hitter"

    # Handle duplicate names (keep first occurrence)
    duplicates = df[df["Name"].duplicated(keep="first")]["Name"].unique()
    if len(duplicates) > 0:
        print(
            f"  Note: Dropping {len(duplicates)} duplicate hitter names: {list(duplicates)[:5]}..."
        )
        df = df.drop_duplicates(subset="Name", keep="first")

    # Select and validate required columns
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

    # Add WAR column if available (for generic player value)
    if "WAR" in df.columns:
        df["WAR"] = df["WAR"].fillna(0.0)
    else:
        df["WAR"] = 0.0

    # Preserve MLBAMID if it exists (needed for MLB API age lookups)
    if "MLBAMID" not in df.columns:
        df["MLBAMID"] = None

    print(
        f"Loaded {len(df)} hitter projections ({n_with_pos} with positions from DB, {n_dh_default} defaulted to DH)"
    )

    # Include WAR and MLBAMID in output
    return_cols = required_cols + ["WAR", "MLBAMID"]
    return df[return_cols].copy()


def load_pitcher_projections(filepath: str) -> pd.DataFrame:
    """
    Load pitcher projections from FanGraphs CSV.

    Args:
        filepath: Path to pitcher projections CSV

    Returns:
        DataFrame with columns:
            Name (with -P suffix), Team, Position, IP, W, SV, K, ERA, WHIP,
            player_type='pitcher', WAR, MLBAMID
    """
    df = pd.read_csv(filepath)

    # Append -P suffix to all names
    df["Name"] = df["Name"].astype(str) + "-P"

    # Fill null Team with 'FA' (free agent)
    df["Team"] = df["Team"].fillna("FA")

    # Rename SO -> K
    df = df.rename(columns={"SO": "K"})

    # Position = 'SP' if GS >= 3 else 'RP'
    df["Position"] = np.where(df["GS"] >= 3, "SP", "RP")

    n_sp = (df["Position"] == "SP").sum()
    n_rp = (df["Position"] == "RP").sum()

    # Add player_type
    df["player_type"] = "pitcher"

    # Handle duplicate names (keep first occurrence)
    duplicates = df[df["Name"].duplicated(keep="first")]["Name"].unique()
    if len(duplicates) > 0:
        print(
            f"  Note: Dropping {len(duplicates)} duplicate pitcher names: {list(duplicates)[:5]}..."
        )
        df = df.drop_duplicates(subset="Name", keep="first")

    # Select and validate required columns
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

    # Add WAR column if available (for generic player value)
    if "WAR" in df.columns:
        df["WAR"] = df["WAR"].fillna(0.0)
    else:
        df["WAR"] = 0.0

    # Preserve MLBAMID if it exists (needed for MLB API age lookups)
    if "MLBAMID" not in df.columns:
        df["MLBAMID"] = None

    print(f"Loaded {len(df)} pitcher projections ({n_sp} SP, {n_rp} RP)")

    # Include WAR and MLBAMID in output
    return_cols = required_cols + ["WAR", "MLBAMID"]
    return df[return_cols].copy()


def load_projections(
    hitter_path: str | None = None,
    pitcher_path: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    Load and combine hitter and pitcher projections.

    Args:
        hitter_path: Path to hitter projections CSV (if None, uses config)
        pitcher_path: Path to pitcher projections CSV (if None, uses config)
        db_path: Optional path to external position database

    Returns:
        Combined DataFrame with columns:
            Name, Team, Position, player_type,
            PA, R, HR, RBI, SB, OPS,    (hitting - 0 for pitchers)
            IP, W, SV, K, ERA, WHIP,    (pitching - 0 for hitters)
            WAR, MLBAMID

        Two-way players (e.g., Ohtani) appear as separate rows:
        - "Shohei Ohtani-H" with hitting stats, pitching cols = 0
        - "Shohei Ohtani-P" with pitching stats, hitting cols = 0
    """
    # Use config defaults if paths not provided
    if hitter_path is None or pitcher_path is None:
        from .config import HITTER_PROJ_PATH, PITCHER_PROJ_PATH

        if hitter_path is None:
            hitter_path = HITTER_PROJ_PATH
        if pitcher_path is None:
            pitcher_path = PITCHER_PROJ_PATH

    positions = load_positions_from_db(db_path) if db_path else {}
    hitters = load_hitter_projections(hitter_path, positions)
    pitchers = load_pitcher_projections(pitcher_path)

    # Define all stat columns
    hitting_stat_cols = ["PA", "R", "HR", "RBI", "SB", "OPS"]
    pitching_stat_cols = ["IP", "W", "SV", "K", "ERA", "WHIP"]

    # Add missing columns with zeros
    for col in pitching_stat_cols:
        hitters[col] = 0.0
    for col in hitting_stat_cols:
        pitchers[col] = 0.0

    # Align column order (include WAR and MLBAMID)
    all_cols = (
        ["Name", "Team", "Position", "player_type"]
        + hitting_stat_cols
        + pitching_stat_cols
        + ["WAR", "MLBAMID"]
    )
    hitters = hitters[all_cols]
    pitchers = pitchers[all_cols]

    # Concatenate
    combined = pd.concat([hitters, pitchers], ignore_index=True)

    # Compute SGP value for each player
    combined["SGP"] = combined.apply(compute_sgp_value, axis=1)

    print(
        f"Combined projections: {len(combined)} total players "
        f"({len(hitters)} hitters, {len(pitchers)} pitchers)"
    )
    print(
        f"SGP range: {combined['SGP'].min():.1f} to {combined['SGP'].max():.1f} "
        f"(mean: {combined['SGP'].mean():.1f})"
    )

    return combined


# === FANTRAX ROSTER CONVERSION ===


def parse_fantrax_roster(filepath: str) -> pd.DataFrame:
    """
    Parse a single Fantrax roster CSV file.

    Handles the multi-section format (separate Hitting/Pitching sections).

    Returns:
        DataFrame with columns: name, team, player_type, status
        - name: Player name with -H/-P suffix based on section
        - team: MLB team abbreviation
        - player_type: 'hitter' or 'pitcher' (from section)
        - status: 'active', 'prospect', or 'IR'
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    current_section = None
    headers = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for section headers
        if ',"Hitting"' in line or line == '"","Hitting"':
            current_section = "hitter"
            headers = None
            continue
        elif ',"Pitching"' in line or line == '"","Pitching"':
            current_section = "pitcher"
            headers = None
            continue

        # Parse CSV line manually (simple case)
        parts = []
        in_quote = False
        current = ""
        for char in line:
            if char == '"':
                in_quote = not in_quote
            elif char == "," and not in_quote:
                parts.append(current.strip('"'))
                current = ""
            else:
                current += char
        parts.append(current.strip('"'))

        # Header row detection
        if parts[0] == "ID" or parts[1] == "Pos":
            headers = parts
            continue

        # Skip if no section active or no headers
        if current_section is None or headers is None:
            continue

        # Parse data row
        row_dict = {h: v for h, v in zip(headers, parts)}

        player_name = row_dict.get("Player", "")
        team = row_dict.get("Team", "")
        status = row_dict.get("Status", "Act")

        if not player_name:
            continue

        # Add suffix based on section
        suffix = "-H" if current_section == "hitter" else "-P"
        name_with_suffix = player_name + suffix

        records.append(
            {
                "name": name_with_suffix,
                "team": team,
                "player_type": current_section,
                "status": status,
            }
        )

    return pd.DataFrame(records)


def convert_fantrax_rosters_from_dir(
    raw_rosters_dir: str,
    my_team_filename: str = "my_team.csv",
    output_dir: str | None = None,
) -> tuple[str, str]:
    """
    Convert all Fantrax rosters in a directory to pipeline format.

    Args:
        raw_rosters_dir: Directory containing Fantrax CSV files
        my_team_filename: Filename of my team's roster
        output_dir: Where to write output files (default: parent of raw_rosters_dir)

    Returns:
        Tuple of (my_roster_path, opponent_rosters_path)
    """
    raw_dir = Path(raw_rosters_dir)
    assert raw_dir.exists(), f"Raw rosters directory not found: {raw_rosters_dir}"

    # Find all CSV files
    csv_files = list(raw_dir.glob("*.csv"))
    assert len(csv_files) > 0, f"No CSV files found in {raw_rosters_dir}"

    print(f"Converting {len(csv_files)} roster files from {raw_rosters_dir}...")

    # Separate my team from opponents
    my_team_path = raw_dir / my_team_filename
    assert my_team_path.exists(), f"My team file not found: {my_team_path}"

    opponent_files = [f for f in csv_files if f.name != my_team_filename]
    opponent_files.sort()  # Alphabetical order

    print(f"  My team: {my_team_filename}")
    print(f"  Opponents: {[f.name for f in opponent_files]}")

    # Parse my team
    my_roster_df = parse_fantrax_roster(str(my_team_path))

    # Parse opponent rosters with team_id assignment
    opponent_records = []
    for idx, opp_file in enumerate(opponent_files, start=1):
        opp_df = parse_fantrax_roster(str(opp_file))
        opp_df["team_id"] = idx
        opponent_records.append(opp_df)

    opponent_roster_df = pd.concat(opponent_records, ignore_index=True)

    # Determine output directory
    if output_dir is None:
        output_dir = raw_dir.parent
    else:
        output_dir = Path(output_dir)

    # Write output files
    my_roster_path = output_dir / "my-roster.csv"
    opponent_rosters_path = output_dir / "opponent-rosters.csv"

    my_roster_df.to_csv(my_roster_path, index=False)
    opponent_roster_df.to_csv(opponent_rosters_path, index=False)

    n_active = (my_roster_df["status"] == "active").sum()
    n_inactive = len(my_roster_df) - n_active
    print(f"Wrote my-roster.csv ({n_active} active, {n_inactive} inactive)")
    print(
        f"Wrote opponent-rosters.csv ({len(opponent_roster_df)} total players across {len(opponent_files)} teams)"
    )

    return str(my_roster_path), str(opponent_rosters_path)


# === NAME MATCHING AND CORRECTION ===


def normalize_name(name: str) -> str:
    """
    Normalize player name for fuzzy comparison.

    CRITICAL: Preserves -H/-P suffix!

    Handles:
        - Accented characters (Rodríguez → rodriguez)
        - Name suffixes like Jr., Sr. (removed)
    """
    # Preserve suffix
    suffix = ""
    if name.endswith("-H"):
        suffix = "-H"
        name = name[:-2]
    elif name.endswith("-P"):
        suffix = "-P"
        name = name[:-2]

    # Normalize unicode (decompose accented characters)
    name = unicodedata.normalize("NFD", name)
    # Remove combining characters (accents)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")

    # Lowercase
    name = name.lower()

    # Remove suffixes
    for suffix_remove in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        name = name.replace(suffix_remove, "")

    # Normalize apostrophes and special chars
    name = name.replace("\u2019", "'").replace("`", "'")

    return name.strip() + suffix.lower()


def _compute_similarity(s1: str, s2: str) -> float:
    """Compute simple similarity ratio between two strings."""
    # Normalize for comparison
    n1 = normalize_name(s1)
    n2 = normalize_name(s2)

    if n1 == n2:
        return 1.0

    # Simple character overlap ratio
    set1 = set(n1)
    set2 = set(n2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def find_name_mismatches(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
) -> pd.DataFrame:
    """
    Find roster names that don't match projections and suggest corrections.

    Returns:
        DataFrame with columns: roster_name, suggested_match, similarity_score
        Only includes names without exact matches.
    """
    roster_df = pd.read_csv(roster_path)
    roster_names = set(roster_df["name"].unique())
    proj_names = set(projections["Name"].unique())

    # Find unmatched
    unmatched = roster_names - proj_names

    if len(unmatched) == 0:
        return pd.DataFrame(
            columns=["roster_name", "suggested_match", "similarity_score"]
        )

    # Find suggestions
    records = []
    for roster_name in unmatched:
        best_match = None
        best_score = 0.0

        # Only compare with same suffix type
        suffix = roster_name[-2:] if roster_name[-2:] in ["-H", "-P"] else ""
        candidates = [p for p in proj_names if p.endswith(suffix)]

        for proj_name in candidates:
            score = _compute_similarity(roster_name, proj_name)
            if score > best_score:
                best_score = score
                best_match = proj_name

        records.append(
            {
                "roster_name": roster_name,
                "suggested_match": best_match,
                "similarity_score": best_score,
            }
        )

    return pd.DataFrame(records).sort_values("similarity_score", ascending=False)


# Known corrections (hardcoded for common issues)
KNOWN_CORRECTIONS = {
    "Logan OHoppe-H": "Logan O'Hoppe-H",
    "Leodalis De Vries-H": "Leo De Vries-H",
    "Leodalis De Vries-P": "Leo De Vries-P",
}


def apply_name_corrections(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
    min_similarity: float = 0.9,
) -> None:
    """
    Apply automatic name corrections to a roster file.

    Uses both name AND team to match players, since different players
    can have the same name (e.g., José Ramírez vs Jose Ramirez).

    High-confidence matches (>= min_similarity) are applied automatically.
    Lower-confidence matches are printed for manual review.
    """
    roster_df = pd.read_csv(roster_path)
    proj_names = set(projections["Name"].unique())

    corrections_made = 0
    manual_review = []

    # Build lookup by (normalized_name, team) -> projection_name
    # This handles cases where multiple players have the same normalized name
    proj_by_name_team = {}
    for _, row in projections.iterrows():
        norm_name = normalize_name(row["Name"])
        team = row["Team"]
        key = (norm_name, team)
        # If there's a collision, prefer the one with accents (more specific)
        if key not in proj_by_name_team or len(row["Name"]) > len(
            proj_by_name_team[key]
        ):
            proj_by_name_team[key] = row["Name"]

    # Also build normalized lookup without team (fallback)
    proj_normalized = {normalize_name(n): n for n in proj_names}

    for idx, row in roster_df.iterrows():
        roster_name = row["name"]
        roster_team = row.get("team", "")

        # Check if name matches exactly AND team matches
        # Don't skip if the name exists but for a different team (e.g., Jose Ramirez DET vs CLE)
        if roster_name in proj_names:
            # Verify the team matches
            proj_row = projections[projections["Name"] == roster_name]
            if len(proj_row) > 0 and proj_row.iloc[0]["Team"] == roster_team:
                continue  # Exact match with correct team, skip
            # Name exists but wrong team - try to find the right player
            normalized = normalize_name(roster_name)
            key_with_team = (normalized, roster_team)
            if key_with_team in proj_by_name_team:
                new_name = proj_by_name_team[key_with_team]
                if new_name != roster_name:
                    roster_df.at[idx, "name"] = new_name
                    corrections_made += 1
                    print(
                        f"  Team mismatch fix: {roster_name} ({roster_team}) -> {new_name}"
                    )
            continue

        # Check known corrections first
        if roster_name in KNOWN_CORRECTIONS:
            new_name = KNOWN_CORRECTIONS[roster_name]
            if new_name in proj_names:
                roster_df.at[idx, "name"] = new_name
                corrections_made += 1
                continue

        # Try normalized matching WITH team (preferred - handles Jose Ramirez cases)
        normalized = normalize_name(roster_name)
        key_with_team = (normalized, roster_team)
        if key_with_team in proj_by_name_team:
            new_name = proj_by_name_team[key_with_team]
            if new_name != roster_name:  # Only count if actually changed
                roster_df.at[idx, "name"] = new_name
                corrections_made += 1
            continue

        # Fall back to normalized matching WITHOUT team
        if normalized in proj_normalized:
            new_name = proj_normalized[normalized]
            roster_df.at[idx, "name"] = new_name
            corrections_made += 1
            continue

        # Find best match by similarity
        suffix = roster_name[-2:] if roster_name[-2:] in ["-H", "-P"] else ""
        candidates = [p for p in proj_names if p.endswith(suffix)]

        best_match = None
        best_score = 0.0
        for proj_name in candidates:
            score = _compute_similarity(roster_name, proj_name)
            if score > best_score:
                best_score = score
                best_match = proj_name

        if best_score >= min_similarity and best_match:
            roster_df.at[idx, "name"] = best_match
            corrections_made += 1
        elif best_match:
            manual_review.append((roster_name, best_match, best_score))

    # Write updated file
    roster_df.to_csv(roster_path, index=False)
    print(f"Applied {corrections_made} automatic corrections to {roster_path}")

    if manual_review:
        print(f"*** MANUAL REVIEW NEEDED ({len(manual_review)} names) ***")
        for roster_name, suggestion, score in manual_review:
            print(f"  {roster_name} -> {suggestion} (similarity: {score:.2f})")


# === ROSTER LOADING ===


def load_my_roster(
    filepath: str, projections: pd.DataFrame
) -> tuple[set[str], set[str]]:
    """
    Load my roster from CSV (assumes name corrections already applied).

    Returns:
        Tuple of (active_names, inactive_names)
        - active_names: Set of player names on active roster
        - inactive_names: Set of player names on prospect/IR slots
    """
    roster_df = pd.read_csv(filepath)

    active_names = set(roster_df[roster_df["status"] == "active"]["name"])
    inactive_names = set(roster_df[roster_df["status"] != "active"]["name"])

    # Validate all active names exist in projections
    proj_names = set(projections["Name"])
    unmatched = active_names - proj_names

    assert len(unmatched) == 0, (
        f"Found {len(unmatched)} roster names not in projections:\n"
        f"  {sorted(unmatched)}\n"
        f"Run apply_name_corrections() to fix spelling/accent differences, "
        f"or manually update the roster file."
    )

    print(
        f"My roster: {len(active_names)} active, {len(inactive_names)} inactive (prospects/IR)"
    )

    return active_names, inactive_names


def load_opponent_rosters(
    filepath: str, projections: pd.DataFrame
) -> dict[int, set[str]]:
    """
    Load opponent rosters from CSV.

    Returns:
        Dict mapping team_id to set of active player names.
        Example: {1: {'Mike Trout-H', 'Mookie Betts-H', ...}, 2: {...}, ...}
    """
    roster_df = pd.read_csv(filepath)

    # Filter to active only
    active_df = roster_df[roster_df["status"] == "active"]

    # Group by team_id
    opponent_rosters = {}
    team_ids = sorted(active_df["team_id"].unique())

    assert set(team_ids) == set(range(1, NUM_OPPONENTS + 1)), (
        f"Expected team_ids {{1, 2, ..., {NUM_OPPONENTS}}}, got {set(team_ids)}"
    )

    proj_names = set(projections["Name"])
    all_opponent_names = set()

    for team_id in team_ids:
        team_names = set(active_df[active_df["team_id"] == team_id]["name"])
        opponent_rosters[team_id] = team_names

        # Check for duplicates across teams
        duplicates = team_names & all_opponent_names
        assert len(duplicates) == 0, (
            f"Player(s) appear on multiple opponent teams: {sorted(duplicates)}"
        )
        all_opponent_names |= team_names

    # Validate all names exist in projections
    unmatched = all_opponent_names - proj_names
    assert len(unmatched) == 0, (
        f"Found {len(unmatched)} opponent roster names not in projections:\n"
        f"  {sorted(unmatched)}\n"
        f"Run apply_name_corrections() on opponent-rosters.csv"
    )

    roster_counts = [len(opponent_rosters[i]) for i in range(1, NUM_OPPONENTS + 1)]
    print(
        f"Loaded {NUM_OPPONENTS} opponent rosters: {', '.join(map(str, roster_counts))} active players each"
    )

    return opponent_rosters


def load_all_data(
    hitter_proj_path: str,
    pitcher_proj_path: str,
    my_roster_path: str,
    opponent_rosters_path: str,
    db_path: str = "../mlb_player_comps_dashboard/mlb_stats.db",
) -> tuple[pd.DataFrame, set[str], dict[int, set[str]]]:
    """
    Load all data files and validate. Main entry point for data loading.

    Returns:
        Tuple of (projections, my_active_roster, opponent_rosters)
    """
    projections = load_projections(hitter_proj_path, pitcher_proj_path, db_path)
    my_active_roster, _ = load_my_roster(my_roster_path, projections)
    opponent_rosters = load_opponent_rosters(opponent_rosters_path, projections)

    return projections, my_active_roster, opponent_rosters


# === TEAM TOTAL COMPUTATION ===


def compute_team_totals(
    player_names: Iterable[str],
    projections: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute a team's projected totals in all 10 scoring categories.

    Args:
        player_names: Iterable of player names (must include -H/-P suffix)
        projections: Combined projections DataFrame

    Returns:
        Dict mapping category to team total.
        Example: {'R': 823, 'HR': 245, ..., 'ERA': 3.85, ...}

    Note:
        Ratio stats (OPS, ERA, WHIP) are computed as weighted averages,
        NOT simple sums. OPS uses PA weighting; ERA/WHIP use IP weighting.
    """
    player_names = set(player_names)

    # Filter projections to players on team
    team_df = projections[projections["Name"].isin(player_names)]

    # Validate all players found
    found_names = set(team_df["Name"])
    missing = player_names - found_names
    assert len(missing) == 0, f"Players not found in projections: {sorted(missing)}"

    # Separate by player type
    hitters = team_df[team_df["player_type"] == "hitter"]
    pitchers = team_df[team_df["player_type"] == "pitcher"]

    totals = {}

    # Counting stats for hitters: R, HR, RBI, SB
    for cat in ["R", "HR", "RBI", "SB"]:
        totals[cat] = hitters[cat].sum()

    # OPS (weighted average by PA)
    total_pa = hitters["PA"].sum()
    assert total_pa > 0, f"Total PA is 0 — no hitters on team?"
    totals["OPS"] = (hitters["PA"] * hitters["OPS"]).sum() / total_pa

    # Counting stats for pitchers: W, SV, K
    for cat in ["W", "SV", "K"]:
        totals[cat] = pitchers[cat].sum()

    # ERA and WHIP (weighted average by IP)
    total_ip = pitchers["IP"].sum()
    assert total_ip > 0, f"Total IP is 0 — no pitchers on team?"
    totals["ERA"] = (pitchers["IP"] * pitchers["ERA"]).sum() / total_ip
    totals["WHIP"] = (pitchers["IP"] * pitchers["WHIP"]).sum() / total_ip

    return totals


def compute_all_opponent_totals(
    opponent_rosters: dict[int, set[str]],
    projections: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """
    Compute totals for all 6 opponents.

    Returns:
        Dict mapping team_id to their category totals.
        {1: {'R': 800, 'HR': 230, ...}, 2: {...}, ...}
    """
    opponent_totals = {}

    for team_id in tqdm(
        sorted(opponent_rosters.keys()), desc="Computing opponent totals"
    ):
        opponent_totals[team_id] = compute_team_totals(
            opponent_rosters[team_id], projections
        )

    return opponent_totals


# === QUALITY SCORE COMPUTATION ===


def compute_quality_scores(projections: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a scalar quality score for each player.

    Methodology:
        Hitters: Mean z-score of (R, HR, RBI, SB)
                 Do NOT include OPS (ratio stat)

        Pitchers: Mean z-score of (W, SV, K, -ERA, -WHIP)
                  Negate ERA/WHIP before z-scoring

    Returns:
        DataFrame with columns: Name, player_type, quality_score
    """
    hitters = projections[projections["player_type"] == "hitter"].copy()
    pitchers = projections[projections["player_type"] == "pitcher"].copy()

    # Hitter quality: mean z-score of counting stats
    hitter_stats = ["R", "HR", "RBI", "SB"]
    hitter_z_scores = pd.DataFrame()
    for stat in hitter_stats:
        std = hitters[stat].std()
        assert std > MIN_STAT_STANDARD_DEVIATION, (
            f"Standard deviation of {stat} is too low: {std}"
        )
        hitter_z_scores[stat] = (hitters[stat] - hitters[stat].mean()) / std

    hitters["quality_score"] = hitter_z_scores.mean(axis=1)

    # Pitcher quality: mean z-score (negate ERA/WHIP)
    pitchers_z = pd.DataFrame()
    for stat in ["W", "SV", "K"]:
        std = pitchers[stat].std()
        assert std > MIN_STAT_STANDARD_DEVIATION, (
            f"Standard deviation of {stat} is too low: {std}"
        )
        pitchers_z[stat] = (pitchers[stat] - pitchers[stat].mean()) / std

    for stat in ["ERA", "WHIP"]:
        std = pitchers[stat].std()
        assert std > MIN_STAT_STANDARD_DEVIATION, (
            f"Standard deviation of {stat} is too low: {std}"
        )
        # Negate so lower is better -> higher z-score
        pitchers_z[stat] = -(pitchers[stat] - pitchers[stat].mean()) / std

    pitchers["quality_score"] = pitchers_z.mean(axis=1)

    # Combine
    result = pd.concat(
        [
            hitters[["Name", "player_type", "quality_score"]],
            pitchers[["Name", "player_type", "quality_score"]],
        ],
        ignore_index=True,
    )

    return result


# === PROJECTION UNCERTAINTY ESTIMATION ===


def estimate_projection_uncertainty(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> dict[str, float]:
    """
    Estimate the standard deviation of team totals for each category.

    Args:
        my_totals: My team's category totals.
        opponent_totals: Dict mapping opponent_id to their category totals.

    Returns:
        Dict mapping category to standard deviation.
        Example: {'R': 45.2, 'HR': 18.3, ..., 'ERA': 0.35, ...}

    Note:
        This empirical approach captures real league variation.
        The σ_c values are used in the Rosenof (2025) formulation
        for normalizing matchup gaps.
    """
    category_sigmas = {}

    for category in ALL_CATEGORIES:
        # Collect all 7 team values
        values = [my_totals[category]]
        for opp_id in sorted(opponent_totals.keys()):
            values.append(opponent_totals[opp_id][category])

        # Compute standard deviation
        sigma = np.std(values, ddof=0)  # Population std (all teams)

        # Ensure minimum to avoid division by zero
        if sigma < MIN_STAT_STANDARD_DEVIATION:
            sigma = MIN_STAT_STANDARD_DEVIATION

        category_sigmas[category] = sigma

    return category_sigmas
